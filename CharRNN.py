import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils


class CharRNN(nn.Module):

    def __init__(self, vocabulary, embedded_dim = 128, hidden_size = 768, num_layers = 3, dropout = .2):
        super(CharRNN, self).__init__()
        self.embbeded_dim = embedded_dim
        self.vocabulary = vocabulary
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.vocab_size = self.input_size = self.output_size = len(vocabulary)

        self.embedding_layer = nn.Embedding(self.vocab_size, self.embbeded_dim,
                                            padding_idx=vocabulary.pad)
        self.lstm_layer = nn.LSTM(self.input_size, self.hidden_size,
                                  self.num_layers, dropout=self.dropout,
                                  batch_first=True)
        self.linear_layer = nn.Linear(self.hidden_size, self.output_size)
        