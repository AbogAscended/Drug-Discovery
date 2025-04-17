import torch
import torch.nn as nn

class CharRNN(nn.Module):

    def __init__(self, vocab_size, hidden_size = 768):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = int(vocab_size)

        self.linear_encode_in = nn.Linear(self.output_size, self.hidden_size)

        self.gru_encode1 = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.gru_encode2 = nn.GRU(self.hidden_size, self.output_size * 2, batch_first=True)

        self.gru_decode1 = nn.GRU(self.output_size, self.hidden_size, batch_first=True)
        self.gru_decode2 = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)

        self.linear_decode_out = nn.Linear(self.hidden_size, self.output_size)

        self.LReLU = nn.LeakyReLU()
        self.ReLU = nn.ReLU()
        self.tanh = nn.Tanh()

    def encode_network(self, x):
        x = self.linear_encode_in(x)
        x = self.LReLU(x)
        x, _ = self.gru_encode1(x)
        x = self.ReLU(x)
        x, _ = self.gru_encode2(x)
        return x

    def decode_network(self, x):
        x, _ = self.gru_decode1(x)
        x = self.ReLU(x)
        x, _ = self.gru_decode2(x)
        x = self.LReLU(x)
        x = self.linear_decode_out(x)
        x = self.tanh(x)
        return x

    def encode(self, x):
        h = self.encode_network(x)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        std = torch.exp(0.5 * logvar)
        return mu, std

    def reparameterize(self, mu, std):
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, std = self.encode(x)
        z = self.reparameterize(mu, std)
        x_reconstructed = self.decode_network(z)
        return x_reconstructed, mu, std


