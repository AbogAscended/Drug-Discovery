from LossTerm import LossTerm
import torch
from onehotencoder import OneHotEncoder
import numpy as np

tensor = torch.randn(256,59,29)
encoder = OneHotEncoder()
value = LossTerm(tensor).losses()

print(np.log(value))