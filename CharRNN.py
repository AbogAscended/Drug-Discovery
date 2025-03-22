import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils


class CharRNN(nn.Module):

    def __init__(self, vocabulary, embedded_dim = 768, hidden_size = 768, num_layers = 3, dropout = .2):
        super(CharRNN, self).__init__()
        self.embbeded_dim = embedded_dim
        self.vocabulary = vocabulary
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.vocab_size = self.input_size = self.output_size = self.vocabulary

        self.embedding_layer = nn.Embedding(self.vocab_size, self.embbeded_dim, padding_idx=27)
        self.gru = nn.GRU(self.embbeded_dim, self.hidden_size,
                                  self.num_layers, dropout=self.dropout,
                                  batch_first=True)
        self.linear_layer = nn.Linear(self.hidden_size, self.output_size)
        
    def forward(self, x):
        x_indices = torch.argmax(x, dim=2)
        embed = self.embedding_layer(x_indices)
        gru_out, _ = self.gru(embed)
        out = self.linear_layer(gru_out)
        return out