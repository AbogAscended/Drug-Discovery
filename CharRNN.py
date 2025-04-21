import torch
import torch.nn as nn

class CharRNN(nn.Module):

    def __init__(self, vocab_size, hidden_size = 768):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = int(vocab_size)

        self.gru_encode1 = nn.GRU(self.output_size, self.hidden_size, batch_first=True)
        self.gru_encode2 = nn.GRU(self.hidden_size, self.output_size * 2, batch_first=True)

        self.gru_decode1 = nn.GRU(self.output_size, self.hidden_size, batch_first=True)
        self.gru_decode2 = nn.GRU(self.hidden_size, self.output_size, batch_first=True)

        self.LReLU = nn.LeakyReLU()
        self.ReLU = nn.ReLU()
        self.tanh = nn.Tanh()

    def encode_network(self, x, hidden):
        x, hidden = self.gru_encode1(x, hidden)
        x = self.ReLU(x)
        x, hidden = self.gru_encode2(x, hidden)
        x = self.LReLU(x)
        return x, hidden

    def decode_network(self, x, hidden):
        x, hidden = self.gru_decode1(x, hidden)
        x = self.ReLU(x)
        x, hidden = self.gru_decode2(x)
        x = self.tanh(x)
        return x , hidden

    def encode(self, x, hidden):
        h, hidden = self.encode_network(x, hidden)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        std = torch.exp(0.5 * logvar)
        return mu, std, hidden

    def reparameterize(self, mu, std):
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, hidden = None):
        if hidden is None:
            hidden = self.init_hidden(x.size(0)).to(x.device)
        mu, std, hidden = self.encode(x, hidden)
        z = self.reparameterize(mu, std)
        x_reconstructed, hidden = self.decode_network(z, hidden)
        return x_reconstructed, mu, std, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)



