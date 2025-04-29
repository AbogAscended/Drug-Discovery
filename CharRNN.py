import torch
import torch.nn as nn

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

class CharRNNV2(nn.Module):

    def __init__(self, vocab_size, num_layers, n_gram, hidden_size = 768, dropout=0.2):
        super(CharRNNV2, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.n_gram = n_gram
        self.output_size = self.input_size = int(vocab_size)
        self.GRU = nn.GRU(self.output_size*self.n_gram, self.hidden_size, num_layers=self.num_layers,batch_first=True, dropout=dropout)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(self.hidden_size*2, self.output_size)
        self.init_gru_weights()


    def forward(self, x, hidden = None):
        x, hidden = self.GRU(x, hidden)
        final_output = torch.cat((x,hidden[-1].unsqueeze(1).expand(-1, x.shape[1], -1)), dim=-1)
        final_output = self.relu(final_output)
        final_output = self.linear(final_output)
        return final_output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(int(self.num_layers), batch_size, self.hidden_size)

    def init_gru_weights(self):
        if isinstance(self, nn.GRU):
            for name, param in self.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0.0)





