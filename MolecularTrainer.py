#%%
from CharRNN import CharRNN, CharRNNV2
import torch.optim as optim, torch.nn as nn, random, bisect, torch, time, numpy as np, pandas as pd
from onehotencoder import OneHotEncoder
from rdkit import RDLogger, Chem
from torch.utils.data import Dataset, DataLoader, Sampler, get_worker_info
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
#%%
#Basic one hot encoder I made to encode and decode both characters and sequences
endecode = OneHotEncoder()
#Hyperparameters
vocab_size = OneHotEncoder.get_vocab_size(self = endecode)
num_layers = 26
n_gram = 1
dropped_out = 0.3
hidden_size = 1024
learning_rate = 5e-2
num_epochs = 200
batch_size = 128
temp = 1
p = 1
b_start = 0
b_end = 1
anneal_epochs = 20
subset_fraction = 1
sample_size = 1
momentum = .9
weight_decay = 1e-5
#%%
class FileDataset(Dataset):
    def __init__(self, filepaths, encoder, n_gram):
        self.filepaths = filepaths
        self.encoder   = encoder
        self.n_gram    = n_gram

        # build cumulative counts & line‐offset tables as before
        self.counts  = []
        self.offsets = []
        total = 0
        for path in filepaths:
            offs = []
            with open(path, 'rb') as f:
                while True:
                    pos = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    offs.append(pos)
            total += len(offs)
            self.counts.append(total)
            self.offsets.append(offs)

        # placeholder for per‐worker file handles
        self.file_handles = None

    def __len__(self):
        return self.counts[-1]

    def __getitem__(self, idx):
        # ensure worker has opened files
        if self.file_handles is None:
            raise RuntimeError("file_handles not initialized – did you forget worker_init_fn?")

        # map idx → (file_idx, line_idx)
        file_idx = bisect.bisect_right(self.counts, idx)
        prev     = 0 if file_idx == 0 else self.counts[file_idx-1]
        line_idx = idx - prev

        # seek & read from the already-open file handle
        fh = self.file_handles[file_idx]
        fh.seek(self.offsets[file_idx][line_idx])
        seq = fh.readline().decode('utf-8').strip()

        # your n-gram logic
        seq_enc = self.encoder.encode_sequence(seq)  # (L, D)
        L, D    = seq_enc.shape
        n       = self.n_gram

        windows = [seq_enc[i : i + n]       for i in range(L - n)]
        targets = [seq_enc[i + n].view(1, D) for i in range(L - n)]

        return torch.stack(windows), torch.cat(targets, dim=0)

def worker_init_fn(worker_id):
    """
    Opens all files for this worker and stores them on the Dataset.
    """
    worker_info = get_worker_info()
    dataset     = worker_info.dataset  # the FileDataset instance
    # open every file and keep the handle
    dataset.file_handles = [
        open(path, 'rb') for path in dataset.filepaths
    ]

class FileBatchSampler(Sampler):
    def __init__(self, counts, batch_size, shuffle=True, drop_last=True, sample_ratio: float = 1.0):
        self.counts     = counts
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.drop_last  = drop_last
        self.sample_ratio = sample_ratio

        self.batches = []
        prev = 0
        for cum in counts:
            idxs = list(range(prev, cum))
            if shuffle:
                random.shuffle(idxs)
            for i in range(0, len(idxs), batch_size):
                batch = idxs[i : i + batch_size]
                if len(batch) == batch_size or not drop_last:
                    self.batches.append(batch)
            prev = cum

        if shuffle:
            random.shuffle(self.batches)
        if not (0 < sample_ratio <= 1):
            raise ValueError("sample_ratio must be in (0,1]")
        if sample_ratio < 1.0:
            keep_n = int(len(self.batches) * sample_ratio)
            self.batches = random.sample(self.batches, keep_n)

    def __iter__(self):
        yield from self.batches

    def __len__(self):
        return len(self.batches)

filepaths = [f"data/seqs_len{i}.txt" for i in list(range(26,46))]
ds = FileDataset(filepaths, endecode, n_gram=n_gram)
sampler = FileBatchSampler(ds.counts, batch_size=batch_size, shuffle=True ,sample_ratio=sample_size)
loader = DataLoader(ds,
                    batch_sampler=sampler,
                    num_workers=20,
                    worker_init_fn=worker_init_fn
                    )
charRNN = CharRNN(vocab_size, num_layers, n_gram, dropped_out).to(device)
#%%
#Using basic cross-entropy loss
criterion = nn.CrossEntropyLoss(ignore_index=27)

#AdamW
optimizer = optim.AdamW(charRNN.parameters(), lr=learning_rate, amsgrad=True, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=3)
charRNN.train()
hidden = charRNN.init_hidden(batch_size).to(device)
#Typical training loop
print(f'Training for {num_epochs} epochs with {len(loader)} batches of size {batch_size} and {n_gram}-gram encoding')
for epoch in range(num_epochs):
    start_time = time.time()
    total_epoch_loss = 0.0
    if epoch < anneal_epochs:
        current_beta = b_start + (b_end - b_start) * (epoch / anneal_epochs)
    else:
        current_beta = b_end

    line_width = 100
    total_batches = len(loader)


    def log(stage, idx):
        msg = f"Batch {idx + 1}/{total_batches} | {stage}"
        # pad to clear out previous text, then carriage-return & flush
        print(msg.ljust(line_width), end='\r', flush=True)


    for idx, (batch_inputs, batch_targets) in enumerate(loader):
        # — your usual device moves / reshape…
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.squeeze(2).to(device)
        current_batch_size = batch_inputs.size(0)
        seq_len = batch_inputs.size(1)
        batch_inputs = batch_inputs.view(current_batch_size, seq_len, n_gram * vocab_size).to(device)
        target_indices = torch.argmax(batch_targets, dim=2).long().to(device)

        hidden = charRNN.init_hidden(current_batch_size).to(device)

        log("Starting forward", idx)
        logits, mu, std, hidden = charRNN(batch_inputs, hidden)

        log("Finished forward, starting backward", idx)
        logits_permuted = logits.permute(0, 2, 1)
        reconstruction_loss = criterion(logits_permuted, target_indices)
        kl_loss = -0.5 * torch.sum(
            1 + torch.log(std.pow(2) + 1e-8) - mu.pow(2) - std.pow(2),
            dim=1
        ).mean()

        loss = reconstruction_loss + kl_loss * current_beta
        loss.backward()
        log("Finished backward, stepping optimizer", idx)
        nn.utils.clip_grad_norm_(charRNN.parameters(), 5.0)
        optimizer.step()
        optimizer.zero_grad()

        log("Batch complete", idx)
        total_epoch_loss += loss.item()

    # move to next line when epoch is done
    print()

    avg_epoch_loss = total_epoch_loss / len(loader)
    scheduler.step(avg_epoch_loss)

    end_time = time.time()
    epoch_duration = end_time - start_time
    epoch_duration_minutes = int(epoch_duration // 60)
    epoch_duration_seconds = int(epoch_duration % 60)

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_epoch_loss}, Time: {epoch_duration_minutes}m {epoch_duration_seconds}s")
torch.save(charRNN,'Models/charRNNv1-gram.pt')
#%%
#This is a bit wonky as its turning the output into a probability distribution and then takes the smallest group of logits to add up to the probability of top_p then samples those
def top_p_filtering(logits_p, top_p, temp_p):
    probs = nn.functional.softmax(logits_p.squeeze(0)[-1] / temp_p, dim=0)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=0) 
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
    sorted_indices_to_remove[0] = False
    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
    filtered_probs = probs.masked_fill(indices_to_remove, 0).clone()
    filtered_probs = filtered_probs / filtered_probs.sum()
    next_token_idx = torch.multinomial(filtered_probs, 1).item()
    return next_token_idx

def get_compound_token(s, n=n_gram):
    if not isinstance(s, str) or not s or n <= 0:
        return ""

    token_parts = []
    current_length = 0
    string_index = 0

    while current_length < n and string_index < len(s):
        if s[string_index:].startswith('Cl'):
            token_parts.append('Cl')
            current_length += 1
            string_index += 2
        elif s[string_index:].startswith('Br'):
            token_parts.append('Br')
            current_length += 1
            string_index += 2
        else:
            token_parts.append(s[string_index])
            current_length += 1
            string_index += 1

    return "".join(token_parts)

RDLogger.DisableLog('rdApp.*')

def sanitize(smiles: str) -> bool:
    """Return True if `smiles` parses and sanitizes, False otherwise."""
    smi = smiles.strip()
    try:
        mol = Chem.MolFromSmiles(smi, sanitize=True)
        return mol is not None
    except Exception as e:
        print(f"Error sanitizing SMILES '{smi}': {e}")
        return False

def validate_generation(file_path: str) -> float:
    """Return the fraction of lines in file_path that are valid SMILES."""
    valid_count = 0
    total = 0

    with open(file_path, 'r') as f:
        for line in f:
            total += 1
            if sanitize(line):
                valid_count += 1

    if total == 0:
        return 0.0

    return valid_count / total
#%%
charRNN = torch.load('Models/charRNNv1-gram.pt', weights_only=False).to(device)
if n_gram == 1:
    current_n_gram = endecode.encode('[BOS]').to(device)
else:
    string_series = pd.read_csv('data/train.csv', header=None)[0]
    string_series = string_series[string_series.apply(lambda x: isinstance(x,str) and x !='')]
    top_n_grams = string_series.apply(lambda s: get_compound_token(s, n=n_gram-1))
    top_chars = (top_n_grams.value_counts()/sum(top_n_grams.value_counts())).to_dict()
    token = np.random.choice(list(top_chars.keys()),p=list(top_chars.values()))
    start_token = endecode.encode('[BOS]')
    current_n_gram = endecode.encode_sequence(token,skip_append=True)
    current_n_gram = torch.tensor(np.concatenate((start_token,current_n_gram),axis=0)).to(device)

charRNN.to(device)
charRNN.eval()
generations = []
for i in range(int(2e4)):
    generation = []
    charCount = 0
    print(f"Generation {i+1}/{int(2e4)}",end='\r')
    with torch.no_grad():
        hidden = charRNN.init_hidden(1).to(device)
        while True:
            if current_n_gram.dim() == 2:
                current_n_gram = current_n_gram.unsqueeze(0)
            logits, hidden = charRNN(current_n_gram, hidden)
            next_token_index = top_p_filtering(logits, p, temp)
            next_token = torch.zeros(vocab_size)
            next_token[next_token_index] = 1
            char = endecode.decode(next_token)
            charCount += 1
            if char == '[EOS]': break
            generation.append(char)
            current_n_gram = current_n_gram.squeeze(0).to(device)
            next_token = next_token.to(device)
            current_n_gram = torch.concat([current_n_gram[1:],next_token.unsqueeze(0)],dim=0)
    generations.append(''.join(generation))
#%%
with open('data/GRUOnly1P1-gram.txt', 'w') as file:
    for item in generations:
        file.write(f"{item}\n")
#%%
print(f"Valid percentage: {validate_generation('data/GRUOnly1P1-gram.txt')}")