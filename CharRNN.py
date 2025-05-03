import math, torch, torch.nn as nn, torch.nn.functional as F, pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from lightning.pytorch import LightningModule

def reparameterize(mu, std):
    eps = torch.randn_like(std)
    return mu + eps * std


class CharRNN(nn.Module):

    def __init__(self, vocab_size, num_layers, n_gram,dropout=0.2):
        super(CharRNN, self).__init__()
        self.num_layers = num_layers
        self.n_gram = n_gram
        self.output_size = int(vocab_size)
        self.gru_encode = nn.GRU(self.output_size*self.n_gram, self.output_size*2, num_layers=self.num_layers,batch_first=True, dropout=dropout)
        self.gru_decode = nn.GRU(self.output_size, self.output_size*2, num_layers=self.num_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(int(self.output_size*4), self.output_size)
        self.relu = nn.ReLU()
        self.init_gru_weights()

    def encode_network(self, x, hidden):
        x, hidden = self.gru_encode(x, hidden)
        return x, hidden

    def decode_network(self, x, hidden):
        x, hidden = self.gru_decode(x, hidden)
        return x , hidden

    def encode(self, x, hidden):
        h, hidden = self.encode_network(x, hidden)
        mu, log_var = torch.chunk(h, 2, dim=-1)
        std = torch.exp(0.5 * log_var)
        return mu, std, hidden

    def forward(self, x, hidden = None):
        mu, std, hidden = self.encode(x, hidden)
        z = reparameterize(mu, std)
        x_reconstructed, hidden = self.decode_network(z, hidden)
        hidden_final = hidden[-1].unsqueeze(1).expand(-1, x_reconstructed.shape[1], -1)
        combined = torch.cat((x_reconstructed,hidden_final), dim=-1)
        combined = self.relu(combined)
        final_output = self.linear(combined)
        return final_output, mu, std, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(int(self.num_layers), batch_size, self.output_size*2)

    def init_gru_weights(self):
        if isinstance(self, nn.GRU):
            for name, param in self.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0.0)

class CharRNNV2(LightningModule):
    def __init__(self, vocab_size, num_layers, n_gram, total_steps, warmup_steps, lr, hidden_size=1024, dropout=0.2,):
        super().__init__()
        self.save_hyperparameters()
        self.GRU = nn.GRU(
            self.hparams.vocab_size * self.hparams.n_gram,
            self.hparams.hidden_size,
            num_layers=self.hparams.num_layers,
            batch_first=True,
            dropout=self.hparams.dropout,
        )
        self.linear = nn.Linear(self.hparams.hidden_size * 2, self.hparams.vocab_size)
        self._init_gru_weights()

    def forward(self, x, hidden=None):
        x, hidden = self.GRU(x, hidden)
        cat = torch.cat([x,hidden[-1].unsqueeze(1).expand(-1, x.size(1), -1),],dim=-1)
        return self.linear(cat), hidden

    def training_step(self, batch, batch_idx):
        x, y = batch
        B, T, _, _ = x.size()
        x = x.view(B, T, self.hparams.n_gram * self.hparams.vocab_size)
        target = y.argmax(dim=2)

        hidden = self._init_hidden(B, x.device)
        logits, _ = self(x, hidden)

        loss = F.cross_entropy(logits.permute(0, 2, 1), target)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        B, T, _, _ = x.size()
        x = x.view(B, T, self.hparams.n_gram * self.hparams.vocab_size)
        target = y.argmax(dim=2)

        hidden = self._init_hidden(B, x.device)
        logits, _ = self(x, hidden)

        loss = F.cross_entropy(logits.permute(0, 2, 1), target)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = {
            "scheduler": LambdaLR(optimizer, lr_lambda=self._combined_lambda),
            "interval": "step"
        }
        return [optimizer], [scheduler]

    def _combined_lambda(self, step):
        if step < self.hparams.warmup_steps:
            return float(step) / float(max(1, self.hparams.warmup_steps))
        progress = float(step - self.hparams.warmup_steps) / float(max(1,self.hparams.total_steps - self.hparams.warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def _init_hidden(self, batch_size, device):
        return torch.zeros(self.hparams.num_layers, batch_size, self.hparams.hidden_size,device=device)

    def _init_gru_weights(self):
        for name, param in self.GRU.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
