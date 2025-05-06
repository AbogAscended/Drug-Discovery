import math
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from rdkit import Chem, RDLogger
import selfies as sf
import re

# Stop annoying warnings (hopefully not important)
RDLogger.DisableLog('rdApp.*')
# Seeding
seed_everything(4990)

# Tokenization
def tokenize_selfies(s):
    return re.findall(r"\[[^\]]+\]", s)

# Vocab & Dataset using integer sequences ---
class Vocab:
    def __init__(self, selfies_list):
        tokens = set(t for s in selfies_list for t in tokenize_selfies(s))
        self.itos = ['<pad>', '<unk>'] + sorted(tokens)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

class SelfiesDataset(Dataset):
    def __init__(self, smiles_list, vocab, max_len):
        self.vocab = vocab
        pad_id = vocab.stoi['<pad>']
        unk_id = vocab.stoi['<unk>']
        sequences = []
        for smi in smiles_list:
            s = sf.encoder(smi)
            toks = tokenize_selfies(s)
            if 0 < len(toks) <= max_len:
                ids = [vocab.stoi.get(t, unk_id) for t in toks]
                ids += [pad_id] * (max_len - len(ids))
                sequences.append(ids)
        self.data = torch.tensor(sequences, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ids = self.data[idx]
        return ids, ids.clone()

# VAE Model with Embedding & KL
class CharVAELightning(pl.LightningModule):
    def __init__(self, vocab, hidden_size=256, latent_dim=128,
                 num_layers=1, lr=1e-3, total_steps=None,
                 warmup_steps=0, max_len=20):
        super().__init__()
        # save all hyperparameters
        self.save_hyperparameters()
        self.max_len = max_len
        V = len(vocab.itos)
        # Embeddings
        self.embedding = nn.Embedding(num_embeddings=V, embedding_dim=hidden_size)
        # encoder/decoder
        self.encoder = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size, latent_dim)
        self.decoder_in = nn.Linear(latent_dim, hidden_size)
        self.decoder = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.out_proj = nn.Linear(hidden_size, V)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.stoi['<pad>'])
# Encode
    def encode(self, x_ids):
        emb = self.embedding(x_ids)
        _, h = self.encoder(emb)
        h = h[-1]
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std
# Decode
    def decode(self, z, x_ids, temperature=1.0):
        emb_in = self.embedding(x_ids)
        h0 = torch.tanh(self.decoder_in(z)).unsqueeze(0)
        h0 = h0.repeat(self.hparams.num_layers, 1, 1)
        out, _ = self.decoder(emb_in, h0)
        logits = self.out_proj(out)
        return logits / temperature

    def forward(self, x_ids, temperature=1.0):
        mu, logvar = self.encode(x_ids)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z, x_ids, temperature)
        return logits, mu, logvar

    def compute_runlength_penalty(self, logits):
        preds = logits.argmax(dim=-1)
        counts = torch.stack([(preds == i).sum(dim=1) for i in range(logits.size(-1))], dim=1).float()
        max_occ = logits.size(1) // 4
        overuse = (counts - max_occ).clamp(min=0)
        return overuse.sum(dim=1).mean() / max_occ

    def kl_weight(self, step):
        return min(1.0, step / 10000)

    def training_step(self, batch, batch_idx):
        x_ids, y_ids = batch
        logits, mu, logvar = self(x_ids)
        V = logits.size(-1)
        rec = self.loss_fn(logits.view(-1, V), y_ids.view(-1))
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x_ids.size(0)
        runlen = self.compute_runlength_penalty(logits)
        kl_wt = self.kl_weight(self.global_step)
        loss = rec + kl_wt * kl + 0.1 * runlen
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_ids, y_ids = batch
        logits, mu, logvar = self(x_ids)
        V = logits.size(-1)
        rec = self.loss_fn(logits.view(-1, V), y_ids.view(-1))
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x_ids.size(0)
        runlen = self.compute_runlength_penalty(logits)
        kl_wt = self.kl_weight(self.global_step)
        loss = rec + kl_wt * kl + 0.1 * runlen
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        if self.hparams.total_steps:
            def lr_lambda(step):
                prog = max(0, step - self.hparams.warmup_steps)
                return 0.5 * (1 + math.cos(math.pi * prog / (self.hparams.total_steps - self.hparams.warmup_steps)))
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
            return [opt], [{'scheduler': sched, 'interval': 'step'}]
        return opt

# Callback for generation and evaluation
class GenerationCallback(pl.Callback):
    def __init__(self, vocab, training_smiles, sample_n=2000, temperature=1.2):
        super().__init__()
        self.vocab = vocab
        self.training_smiles = training_smiles
        self.sample_n = sample_n
        self.temperature = temperature

    def on_validation_epoch_end(self, trainer, pl_module):
        selfies_preds, smiles_preds = generate_selfies_samples(
            pl_module, self.vocab,
            n=self.sample_n,
            temperature=self.temperature,
            device=pl_module.device
        )
        validity, novelty, _, _ = evaluate_smiles(smiles_preds, self.training_smiles)
        pl_module.log('gen_validity', validity)
        pl_module.log('gen_novelty', novelty)

# Generation and Evaluation
def generate_selfies_samples(model, vocab, n=1000, temperature=1.0, top_k=20, device='cpu'):
    model.eval().to(device)
    z = torch.randn(n, model.hparams.latent_dim, device=device)
    tok0 = torch.full((n,1), vocab.stoi['<pad>'], device=device, dtype=torch.long)
    inp = tok0
    h = torch.tanh(model.decoder_in(z)).unsqueeze(0).repeat(model.hparams.num_layers,1,1)

    seqs = []
    for _ in range(model.max_len):
        emb = model.embedding(inp)
        out, h = model.decoder(emb, h)
        logits = model.out_proj(out[:, -1]) / temperature
        values, _ = torch.topk(logits, top_k)
        min_vals = values[:, -1].unsqueeze(1)
        logits[logits < min_vals] = -float('Inf')
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1)
        seqs.append(nxt)
        inp = nxt

    seqs = torch.cat(seqs, dim=1)
    selfies_strings = [''.join(vocab.itos[i] for i in seq.squeeze() if i>1) for seq in seqs]
    smiles = []
    for s in selfies_strings:
        try:
            smiles.append(sf.decoder(s))
        except sf.DecoderError:
            smiles.append("")
    return selfies_strings, smiles

# Evaluate for valid and novel
def evaluate_smiles(generated_smiles, training_smiles):
    valid = []
    for smi in generated_smiles:
        try:
            m = Chem.MolFromSmiles(smi, sanitize=True)
            if m:
                valid.append(smi)
        except:
            pass
    valid_set = set(valid)
    novel_set = valid_set - set(training_smiles)
    validity = 100 * len(valid) / len(generated_smiles)
    novelty = 100 * sum(smi not in training_smiles for smi in valid) / len(valid)
    return validity, novelty, valid_set, novel_set

# Main function - Load, tokenize once, create dataset, train, evaluate
if __name__ == '__main__':
    CSV_PATH = '/content/train.txt'
    SAMPLE_SIZE = 300000
    BATCH_SIZE = 128
    MAX_EPOCHS = 3
    LR = 1e-3
    LATENT_DIM = 64
    NUM_WORKERS = 4

    # Load SMILES
    smiles_list = []
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= SAMPLE_SIZE:
                break
            smi = row['SMILES'].strip()
            if smi:
                smiles_list.append(smi)
    print(f"Loaded {len(smiles_list)} SMILES")

    # Encode to SELFIES for vocab
    selfies_list = [sf.encoder(smi) for smi in smiles_list]
    max_len = max(len(tokenize_selfies(s)) for s in selfies_list)
    vocab = Vocab(selfies_list)
    print(f"Vocab size: {len(vocab.itos)}, max_len: {max_len}")

    # Dataset & DataLoaders
    dataset = SelfiesDataset(smiles_list, vocab, max_len)
    train_len = int(0.9 * len(dataset))
    val_len = len(dataset) - train_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])
    val_ds, _ = random_split(val_ds, [min(5000, len(val_ds)), max(0, len(val_ds)-5000)])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            num_workers=NUM_WORKERS, pin_memory=True,
                            persistent_workers=True)
    print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")

    # Model + Trainer + Callbacks
    total_steps = len(train_loader) * MAX_EPOCHS
    model = CharVAELightning(
        vocab=vocab,
        hidden_size=512,
        latent_dim=LATENT_DIM,
        num_layers=2,
        lr=LR,
        total_steps=total_steps,
        warmup_steps=int(0.1 * total_steps),
        max_len=max_len
    )
    gen_cb = GenerationCallback(vocab, smiles_list, sample_n=2000, temperature=1.2)
    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision=16,
        callbacks=[
            pl.callbacks.ModelCheckpoint(monitor='train_loss'),
            pl.callbacks.EarlyStopping(monitor='train_loss', patience=3),
            gen_cb
        ],
        profiler='simple'
    )

    # Fit model
    trainer.fit(model, train_loader, val_loader)

    # Final generation & evaluation
    selfies_preds, smiles_preds = generate_selfies_samples(
        model, vocab, n=2000, temperature=1.2, device=model.device
    )
    validity, novelty, valid_set, novel_set = evaluate_smiles(smiles_preds, smiles_list)

    print(f"\nGenerated {len(smiles_preds)} samples")
    print(f"Validity: {validity:.2f}%")
    print(f"Novelty:  {novelty:.2f}%\n")

    print("Distinct Valid SMILES:")
    for smi in sorted(valid_set):
        print(smi)
    print(f"\n# Distinct Valid: {len(valid_set)}\n")

    print("Distinct Novel SMILES:")
    for smi in sorted(novel_set):
        print(smi)
    print(f"\n# Distinct Novel: {len(novel_set)}")