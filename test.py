import numpy as np
import torch
from lossTerm import lossTerm
from onehotencoder import onehotencoder

#Loss Term Values
# Value added for failing sanitization # invalid molecule
fail_sanitize = 0.5
# Value added for too many duplicates of a character in a sequence
too_many_duplicates = 0.2
# Value added for invalid characters like '[UNK]
wrong_value = 0.2
# Value added for each swap that fails to sanitize
fail_swaps = 0.1

sequences = ['CCCS(=O)c1ccc2[nH]c(=NC(=O)OC)[nH]c2c1',
'CC(C)(C)C(=O)C(Oc1ccc(Cl)cc1)n1ccnc1',
'Cc1c(Cl)cccc1Nc1ncccc1C(=O)OCC(O)CO',
'Cn1cnc2c1c(=O)n(CC(O)CO)c(=O)n2C',
'CC1Oc2ccc(Cl)cc2N(CC(O)CO)C1=O',
'CCOC(=O)c1cncn1C1CCCc2ccccc21']

# Convert the sequences to tensors (one hot encoded) using stack
onehot = onehotencoder()  # Ensure the onehotencoder class is implemented and has an encode_sequence method
logits = []
encoded_strings = [onehot.encode_sequence(seq, targets=False).numpy() for seq in sequences]
tensors = [torch.tensor(enc) for enc in encoded_strings]

logits = torch.stack(tensors)

# Create a lossTerm object
lossTerm_obj = lossTerm(logits.clone().detach(), len(sequences), fail_sanitize, too_many_duplicates, wrong_value, fail_swaps).losses()  # Ensure the lossTerm class is implemented and has a losses method
# ????????????????????????????????????????
