import torch
import torch.nn.functional as F


class OneHotEncoder:
    def __init__(self):
        self.characters = ['Br', 'N', ')', 'c', 'o', '6', 's', 'Cl', '=', '2', ']', 'C', 'n', 'O', '4', '1', '#', 'S', 'F', '3', '[', '5', 'H', '(', '-', '[BOS]', '[EOS]', '[UNK]', '[PAD]']
        self.cti = {char: idx for idx, char in enumerate(self.characters)}
        self.itc = {idx: char for char, idx in self.cti.items()}
        self.len = len(self.cti)

    def encode(self, char):
        return F.one_hot(torch.tensor([self.cti.get(char)]), num_classes = self.len).float()
    
    def decode(self, vec):
        return self.itc[torch.argmax(vec).item()]
    
    def get_vocab_size(self):
        return len(self.characters)
    
    def encode_sequence(self, sequence, targets = False, skip_append = False):
        sequence = sequence.strip()
        tokens = []
        i = 0
        if not (skip_append or targets):
            tokens.append('[BOS]')
        while i < len(sequence):
            if i + 1 < len(sequence) and sequence[i:i + 2] == 'Cl':
                tokens.append('Cl')
                i += 2
            elif i + 1 < len(sequence) and sequence[i:i + 2] == 'Br':
                tokens.append('Br')
                i += 2
            elif i + 4 < len(sequence) and sequence[i:i + 5] == '[PAD]':
                tokens.append('[PAD]')
                i += 5
            elif i + 4 < len(sequence) and sequence[i:i + 5] == '[UNK]':
                tokens.append('[PAD]')
                i += 5
            else:
                tokens.append(sequence[i])
                i += 1
        if not skip_append:
            tokens.append('[EOS]')
            while len(tokens) < 59: tokens.append('[PAD]')

        indices = [self.cti.get(char) for char in tokens]
        return F.one_hot(torch.tensor(indices), num_classes=self.len).float()

    # Takes in a matrix of one-hot encoded vectors and returns a single string.
    def decode_sequence(self, onehot_sequence):
        indices = torch.argmax(onehot_sequence, dim=1).tolist()
        return ''.join([self.itc[idx] for idx in indices])