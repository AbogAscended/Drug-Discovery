import torch
import torch.nn as nn

class CharRNN(nn.Module):

    def __init__(self, vocab_size, num_layers, n_gram):
        super(CharRNN, self).__init__()
        self.num_layers = num_layers
        self.n_gram = n_gram
        self.output_size = int(vocab_size)
        self.gru_encode = nn.GRU(self.output_size, self.output_size*2, num_layers=self.num_layers,batch_first=True)
        self.gru_decode = nn.GRU(self.output_size, self.output_size*2, num_layers=self.num_layers, batch_first=True)
        self.linear = nn.Linear(int(self.n_gram*self.output_size*2 + self.num_layers*self.output_size*2), self.output_size)
        self.LReLU = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.init_gru_weights()

    def encode_network(self, x, hidden):
        x, hidden = self.gru_encode(x, hidden)
        return x, hidden

    def decode_network(self, x, hidden):
        x, hidden = self.gru_decode(x, hidden)
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
        mu, std, hidden = self.encode(x, hidden)
        z = self.reparameterize(mu, std)
        x_reconstructed, hidden = self.decode_network(z, hidden)
        hidden_copy = hidden.clone().view(1,-1)
        x_reconstructed = x_reconstructed.view(1,-1)
        combined = torch.cat((x_reconstructed, hidden_copy), dim=-1)
        combined = self.LReLU(combined)
        final_output = self.linear(combined)
        logits = final_output
        return logits, mu, std, hidden

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



