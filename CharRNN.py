import torch
import torch.nn as nn


class GRUWrapper(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.gru = nn.GRU(*args, **kwargs)

    def forward(self, x):
        output, _ = self.gru(x)
        return output

class CharRNN(nn.Module):

    def __init__(self, vocab_size, pad_idx, hidden_size = 768, num_layers = 3, dropout = .2):
        super(CharRNN, self).__init__()
        self.pad_idx = pad_idx
        self.vocabulary = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.output_size = int(vocab_size)

        self.encoder = nn.Sequential(
            nn.Linear(self.output_size, self.hidden_size),
            nn.LeakyReLU(),
            GRUWrapper(self.hidden_size, self.hidden_size, batch_first=True),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.output_size * 2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.output_size, self.hidden_size),
            nn.LeakyReLU(),
            GRUWrapper(self.hidden_size, self.hidden_size, batch_first=True),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.output_size)
        )

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        std = torch.exp(0.5 * logvar)
        return mu, std

    def reparameterize(self, mu, std):
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, std = self.encode(x)
        z = self.reparameterize(mu, std)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, std


