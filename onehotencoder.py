import torch
import torch.nn.functional as F
class onehotencoder:
    def __init__(self):
        self.characters = ['C', 'c', 'N', 'n', 'O', 'o', 'S', 's', 'P', 'p', 'F', 'f', 'Cl', 'Br', 'I', 'B', 'b', 'H', 'h', '[', ']', '(', ')', '=', '#', '+', '-', '.', '/', '@', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '%', ':', ',', ';','$', '&', '*', '[BOS]','[EOS]','[PAD]']
        self.cti = {char: idx for idx, char in enumerate(self.characters)}
        self.itc = {idx: char for char, idx in self.cti.items()}
        self.len = len(self.cti)

    def encode(self, char):
        return F.one_hot(torch.tensor([self.cti.get(char)]), num_classes = self.len)
    
    def decode(self, vec):
        return self.itc[torch.argmax(vec).item()]
    
    def encode_sequence(self, sequence, targets = False):
        sequence = sequence.strip()
        tokens = []
        if targets == False:
            tokens.append('[BOS]')

        i = 0
        while i < len(sequence):
            if i+1 < len(sequence) and sequence[i:i+2] == 'Cl':
                tokens.append('Cl')
                i += 2
            elif i+1 < len(sequence) and sequence[i:i+2] == 'Br':
                tokens.append('Br')
                i += 2
            elif i+4 < len(sequence) and sequence[i:i+5] == '[PAD]':
                tokens.append('[PAD]')
                i += 5
            else:
                tokens.append(sequence[i])
                i += 1
        tokens.append('[EOS]')
        while len(tokens)<59: tokens.append('[PAD]')
        indices = [self.cti.get(char) for char in tokens]
        return F.one_hot(torch.tensor(indices), num_classes=self.len)

    def decode_sequence(self, onehot_sequence):
        indices = torch.argmax(onehot_sequence, dim=1).tolist()
        return ''.join([self.itc[idx] for idx in indices])