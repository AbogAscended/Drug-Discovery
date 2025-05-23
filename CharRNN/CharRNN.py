from torch.optim.lr_scheduler import LambdaLR
from lightning.pytorch import LightningModule
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def confidence_penalty(logits, beta=0.1):
    P = F.softmax(logits, dim=-1)
    H = -(P * torch.log(P + 1e-12)).sum(dim=-1)
    return -beta * H.mean()


class CharRNN(LightningModule):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        n_gram: int,
        dropout: float,
        lr: float,
        kl_anneal_epochs: int,
        hidden_size: int,
        warmup_steps: int = None,
        max_steps: int = None,
        embedding_dim: int = 128,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.embedding = nn.Embedding(
            num_embeddings=self.hparams.vocab_size,
            embedding_dim=self.hparams.embedding_dim
        )

        self.gru_encode = nn.GRU(
            input_size=self.hparams.embedding_dim * self.hparams.n_gram,
            hidden_size=self.hparams.hidden_size,
            num_layers=self.hparams.num_layers,
            batch_first=True,
            dropout=self.hparams.dropout,
        )

        self.gru_decode = nn.GRU(
            input_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            num_layers=self.hparams.num_layers,
            batch_first=True,
            dropout=self.hparams.dropout,
        )

        self.linear_mu = nn.Linear(self.hparams.hidden_size, self.hparams.vocab_size)
        self.linear_log_var = nn.Linear(self.hparams.hidden_size, self.hparams.vocab_size)
        self.linear_zh = nn.Linear(self.hparams.vocab_size, self.hparams.hidden_size)
        self.linear_final = nn.Linear(self.hparams.hidden_size * 2, self.hparams.vocab_size)
        self._init_gru_weights()

    def _init_gru_weights(self):
        for gru in (self.gru_encode, self.gru_decode):
            for name, param in gru.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0.0)
        for linear in (self.linear_mu, self.linear_log_var, self.linear_zh, self.linear_final):
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)

    def init_hidden(self, batch_size: int):
        return torch.zeros(
            self.hparams.num_layers,
            batch_size,
            self.hparams.hidden_size,
            device=self.device
        )

    def forward(self, x, hidden=None):
        if x.dtype in (torch.float, torch.double):
            x = x.argmax(dim=-1)
        x = x.long().to(self.device)
        if hidden is None:
            hidden = self.init_hidden(x.size(0))

        B, S, N = x.size()
        x_embed = self.embedding(x)

        x_flat = x_embed.view(B, S, N * self.hparams.embedding_dim)

        enc_out, hidden = self.gru_encode(x_flat, hidden)

        mu = self.linear_mu(enc_out)
        log_var = self.linear_log_var(enc_out)
        std = torch.exp(0.5 * log_var)
        z = mu + std * torch.randn_like(std)

        dec_in = self.linear_zh(z)
        dec_out, hidden = self.gru_decode(dec_in, hidden)

        last_hidden = hidden[-1].unsqueeze(1).expand(-1, dec_out.size(1), -1)
        combined = torch.cat((dec_out, last_hidden), dim=-1)
        logits = self.linear_final(combined)

        return logits, mu, log_var, hidden

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits, mu, log_var, _ = self(x)

        rec_loss = F.cross_entropy(logits.permute(0,2,1), y.argmax(dim=-1)) + confidence_penalty(logits)
        kl_loss = -0.5 * torch.mean(torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1))

        kl_weight = min(1.0, self.current_epoch / self.hparams.kl_anneal_epochs)
        loss = rec_loss + kl_weight * kl_loss

        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_rec', rec_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_kl', kl_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('kl_weight', kl_weight, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits, mu, log_var, _ = self(x)

        val_loss = F.cross_entropy(logits.permute(0,2,1), y.argmax(dim=-1))

        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True)

        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-5)
        scheduler = {
            "scheduler": LambdaLR(optimizer, lr_lambda=self._combined_lambda),
            "interval": "step"
        }
        return [optimizer], [scheduler]

    def _combined_lambda(self, step):
        if step == 0:
            return 1e-8
        if step < self.hparams.warmup_steps:
            return step / float(self.hparams.warmup_steps)
        return (self.hparams.warmup_steps ** 0.5) * (step ** -0.5)


class CharRNNV2(LightningModule):
    def __init__(self, vocab_size, num_layers, n_gram, total_steps=None, warmup_steps=None, lr=None, hidden_size=1024, dropout=0.2, embedding_dim=128):
        super().__init__()
        self.save_hyperparameters()
        self.embedding = nn.Embedding(self.hparams.vocab_size, self.hparams.embedding_dim)
        self.GRU = nn.GRU(
            self.hparams.embedding_dim * self.hparams.n_gram,
            self.hparams.hidden_size,
            num_layers=self.hparams.num_layers,
            batch_first=True,
            dropout=self.hparams.dropout,
        )
        self.linear = nn.Linear(self.hparams.hidden_size * 2, self.hparams.vocab_size)
        self._init_gru_weights()

    def forward(self, x, hidden=None):
        if x.dtype in (torch.float, torch.double):
            x = x.argmax(dim=-1)
        x = x.long().to(self.device)
        if hidden is None:
            hidden = self.init_hidden(x.size(0),device=self.device)
        B, S, N = x.size()
        x_embed = self.embedding(x)
        x_flat = x_embed.view(B, S, N * self.hparams.embedding_dim)
        x, hidden = self.GRU(x_flat, hidden)
        cat = torch.cat([x,hidden[-1].unsqueeze(1).expand(-1, x.size(1), -1),],dim=-1)
        return self.linear(cat), hidden

    def training_step(self, batch, batch_idx):
        x, y = batch
        target = y.argmax(dim=-1)
        hidden = self.init_hidden(x.size(0), x.device)
        logits, _ = self(x, hidden)
        loss = F.cross_entropy(logits.permute(0, 2, 1), target)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        target = y.argmax(dim=-1)
        hidden = self.init_hidden(x.size(0), x.device)
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

    def init_hidden(self, batch_size, device):
        return torch.zeros(self.hparams.num_layers, batch_size, self.hparams.hidden_size,device=device)

    def _init_gru_weights(self):
        for name, param in self.GRU.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
