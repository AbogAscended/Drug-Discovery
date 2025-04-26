#%%
from CharRNN import CharRNN
import torch, torch.optim as optim, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from onehotencoder import OneHotEncoder
from rdkit import Chem
from rdkit import RDLogger
import time
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#%%
#Basic one hot encoder I made to encode and decode both characters and sequences
endecode = OneHotEncoder()
#Hyperparameters
vocab_size = OneHotEncoder.get_vocab_size(self = endecode)
num_layers = 26
n_gram = 1
dropped_out = 0.2
learning_rate = 1e-3
num_epochs = 15
batch_size = 256
temp = 1
p = .95
eps = .01
# LossTerm Variables
fail_sanitize = 0.5
too_many_duplicates = 0.6
wrong_value = 0.1
fail_swaps = 0.1

#%%
#Torch dataset because the processed inputs and outputs were over 60 gb in size

class SequenceDataset(Dataset):
    def __init__(self, file_path, encoder, n_gram = 1):
        self.n_gram = n_gram
        self.file_path = file_path
        self.encoder = encoder
        with open(file_path, 'r') as f:
            self.lines = f.readlines()
    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        input_tensors = []
        target_tensors = []
        sequence = self.lines[idx].strip()
        sequence_input = self.encoder.encode_sequence(sequence)
        sequence_target = self.encoder.encode_sequence(sequence)
        pad = self.encoder.encode('[PAD]').view(1,-1)
        length = sequence_input.shape[0]
        for i in range(length):
            if self.n_gram == 1:
                if length - (i + 1) < self.n_gram:
                    input_tensors.append(sequence_input[i:i+1,:])
                    assert input_tensors[-1].shape[0] == self.n_gram, '1i'
                    target_tensors.append(pad)
                else:
                    input_tensors.append(sequence_input[i:i+1,:])
                    assert input_tensors[-1].shape[0] == self.n_gram, '2i'
                    target_tensors.append(sequence_target[i+1:i+2,:])
            else:
                if length - (i + 1) < self.n_gram:
                    padding = pad.repeat(self.n_gram - length + i,1)
                    input_tensors.append(torch.cat([sequence_input[i:length,:],padding],dim=0))
                    assert input_tensors[-1].shape[0] == self.n_gram, '3i'
                    target_tensors.append(pad)
                else:
                    input_tensors.append(sequence_input[i:i+self.n_gram,:])
                    assert input_tensors[-1].shape[0] == self.n_gram, '4i'
                    target_tensors.append(sequence_target[i+self.n_gram:i+1+self.n_gram,:])
        input_stack = torch.stack(input_tensors)
        target_stack = torch.stack(target_tensors)
        return input_stack, target_stack

#Load the dataset for working
dataset = SequenceDataset('data/train.csv', endecode, n_gram = n_gram)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers= 3)
#%%
#Declare RNN with vocab size, hidden dim size
charRNN = CharRNN(vocab_size, num_layers, n_gram, dropped_out).to(device)

#Using basic cross-entropy loss
criterion = nn.CrossEntropyLoss(ignore_index=28)

#AdamW
optimizer = optim.AdamW(charRNN.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=3)
charRNN.train()
#Typical training loop
for epoch in range(num_epochs):
    start_time = time.time()
    total_epoch_loss = 0.0

    for idx, (batch_inputs, batch_targets) in enumerate(dataloader):
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.squeeze(2).to(device)
        current_batch_size = batch_inputs.size(0)
        seq_len = batch_inputs.size(1)
        batch_inputs = batch_inputs.view(current_batch_size, seq_len, n_gram * vocab_size)
        target_indices = torch.argmax(batch_targets, dim=2).long()

        optimizer.zero_grad()

        hidden = charRNN.init_hidden(current_batch_size).to(device)

        logits, mu, std, hidden = charRNN(batch_inputs, hidden)
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = target_indices.view(-1)

        reconstruction_loss = criterion(logits_flat, targets_flat)
        kl_loss = -0.5 * torch.sum(1 + torch.log(std.pow(2) + 1e-8) - mu.pow(2) - std.pow(2), dim=1)
        kl_loss = torch.mean(kl_loss)
        loss = reconstruction_loss + kl_loss * eps
        loss.backward()

        optimizer.step()
        total_epoch_loss += loss.item()

    avg_epoch_loss = total_epoch_loss / len(dataloader)
    scheduler.step(avg_epoch_loss)

    end_time = time.time()
    epoch_duration = end_time - start_time
    epoch_duration_minutes = int(epoch_duration // 60)
    epoch_duration_seconds = int(epoch_duration % 60)

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_epoch_loss}, Time: {epoch_duration_minutes}m {epoch_duration_seconds}s")

torch.save(charRNN,'Models/charRNN1-gram.pt')
#%%
#This is a bit wonky as its turning the output into a probability distribution and then takes the smallest group of logits to add up to the probability of top_p then samples those
def top_p_filtering(logits_p, top_p, temp_p):
    probs = nn.functional.softmax(logits_p.squeeze(0)[-1] / temp_p, dim=0)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=0) 
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
    sorted_indices_to_remove[0] = False
    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
    filtered_probs = probs.masked_fill(indices_to_remove, 0).clone()
    filtered_probs = filtered_probs / filtered_probs.sum()
    next_token_idx = torch.multinomial(filtered_probs, 1).item()
    return next_token_idx

def get_compound_token(s, n=n_gram):
    if not isinstance(s, str) or not s or n <= 0:
        return ""

    token_parts = []
    current_length = 0
    string_index = 0

    while current_length < n and string_index < len(s):
        if s[string_index:].startswith('Cl'):
            token_parts.append('Cl')
            current_length += 1
            string_index += 2
        elif s[string_index:].startswith('Br'):
            token_parts.append('Br')
            current_length += 1
            string_index += 2
        else:
            token_parts.append(s[string_index])
            current_length += 1
            string_index += 1

    return "".join(token_parts)

def validate_generation(file_path):
    # initialize variables
    valid_count, invalid_count = 0, 0

    # read the lines of the file
    with open(file_path, 'r') as f:
        # read all lines into sequences
        sequences = f.readlines()

    # count the valid sequences and the invalid sequences
    for sequence in sequences:
        valid = sanitize(sequence) # validate
        if valid == 1: # valid
            valid_count += 1
        else: # invalid
            invalid_count += 1

    # Get the percentage of valid VS invalid sequences
    valid_percentage = valid_count / (valid_count + invalid_count)
    return valid_percentage

def sanitize(sequence):
    # Disable all RDKit warnings
    RDLogger.DisableLog('rdApp.*')

    # check sanitizing for the sequence input
    try:
        # attempt to sanitize
        mol = Chem.MolFromSmiles(sequence, sanitize=True)
        if mol:
            return 1  # valid
        else:
            return 0  # invalid
    except Exception as e:
        print(f"Error sanitizing molecule: {e}")
        return 1  # invalid with error
#%%

charRNN = torch.load('Models/charRNN1-gram.pt', weights_only=False)
currenToken = endecode.encode('[BOS]').to(device)
charRNN.to(device)
charRNN.eval()
generations = []
for i in range(int(5e4)):
    generation = []
    with torch.no_grad():
        while True:
            if currenToken.dim() == 2:
                currenToken = currenToken.unsqueeze(0)
            logits = charRNN(currenToken)
            next_token_index = top_p_filtering(logits, p, temp)
            next_token = torch.zeros(vocab_size)
            next_token[next_token_index] = 1
            char = endecode.decode(next_token)
            if char == '[EOS]': break
            generation.append(char)
            currenToken = next_token.unsqueeze(0).unsqueeze(0).to(device)

    generations.append(''.join(generation))
#%%
with open('GRUOnly95P1-gram.txt', 'w') as file:
    for item in generations:
        file.write(f"{item}\n")

#%%
# This function is used to check if the SMILE string validates using sanitize
def sanitize(sequence):
    # Disable all RDKit warnings
    RDLogger.DisableLog('rdApp.*')

    # check sanitize for the sequence input
    try:
        # attempt to sanitize
        mol = Chem.MolFromSmiles(sequence, sanitize=True)
        if mol:
            return 1  # valid
        else:
            return 0  # invalid
    except Exception as e:
        print(f"Error sanitizing molecule: {e}")
        return 1  # invalid with error

# Get the percentage of valid/invalid molecules
def validate_generation(file_path):
    # initialize variables
    valid_count, invalid_count = 0, 0

    # read the lines of the file
    with open(file_path, 'r') as f:
        # read all lines into sequences
        sequences = f.readlines()

    # count the valid sequences and the invalid sequences
    for sequence in sequences:
        valid = sanitize(sequence) # validate
        if valid == 1: # valid
            valid_count += 1
        else: # invalid
            invalid_count += 1

    # Get the percentage of valid VS invalid sequences
    valid_percentage = valid_count / (valid_count + invalid_count)
    return valid_percentage