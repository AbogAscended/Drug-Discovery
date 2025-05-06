import torch, torch.nn as nn, numpy as np, pandas as pd
from rdkit import RDLogger, Chem

def sanitize(smiles: str) -> bool:
    smi = smiles.strip()
    try:
        mol = Chem.MolFromSmiles(smi, sanitize=True)
        return mol is not None
    except Exception as e:
        print(f"Error sanitizing SMILES '{smi}': {e}")
        return False


class Validator:
    def __init__(self, file_path: str, valid_path: str):
        self.valid_path = valid_path
        self.file_path = file_path
        RDLogger.DisableLog('rdApp.*')

    def validate_generation(self) -> None:
        valid_count = 0
        total = 0
        valid = []
        with open(self.file_path, 'r') as f:
            for line in f:
                total += 1
                if sanitize(line) :
                    valid_count += 1
                    if line.strip() not in valid and line.strip() != '':
                        valid.append(line.strip())
        print(f"Valid %: {valid_count/total * 100}")
        with open(self.valid_path, 'w') as file:
            for item in valid:
                file.write(f"{item.strip()}\n")

    def generate_stats(self):
        with open('data/train.csv', 'r') as f:
            original_lines = f.read().splitlines()
        with open(self.valid_path, 'r') as f:
            valid_lines = f.read().splitlines()
        with open(self.file_path, 'r') as f:
            generated_lines = f.read().splitlines()
        actual_unique = len(valid_lines)/len(generated_lines) * 100
        novel = 0
        for line in valid_lines:
            if line not in original_lines:
                novel += 1
        novel_rate = novel/len(valid_lines) * 100

        return actual_unique, novel_rate
class Generator:
    def __init__(self, char_rnn, endecode, vocab_size, n_gram, p, temp):
        self.charRNN = char_rnn.eval()
        self.endecode = endecode
        self.vocab_size = vocab_size
        self.n_gram = n_gram
        self.p = p
        self.temp = temp
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def top_p_filtering(self, logits_p):
        probs = nn.functional.softmax(logits_p.squeeze(0)[-1] / self.p, dim=0)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=0)
        sorted_indices_to_remove = cumulative_probs > self.p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False
        indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
        filtered_probs = probs.masked_fill(indices_to_remove, 0).clone()
        filtered_probs = filtered_probs / filtered_probs.sum()
        next_token_idx = torch.multinomial(filtered_probs, 1).item()
        return next_token_idx

    def get_compound_token(self,s):
        n = self.n_gram - 1
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

    def generate(self, filepath, amount):
        if self.n_gram == 1:
            current_n_gram = self.endecode.encode('[BOS]').to(self.device)
        else:
            string_series = pd.read_csv('data/train.csv', header=None)[0]
            string_series = string_series[string_series.apply(lambda x: isinstance(x, str) and x != '')]
            top_n_grams = string_series.apply(lambda s: self.get_compound_token(s))
            top_chars = (top_n_grams.value_counts() / sum(top_n_grams.value_counts())).to_dict()
            token = np.random.choice(list(top_chars.keys()), p=list(top_chars.values()))
            start_token = self.endecode.encode('[BOS]')
            current_n_gram = self.endecode.encode_sequence(token, skip_append=True)
            current_n_gram = torch.tensor(np.concatenate((start_token, current_n_gram), axis=0)).to(self.device)

        self.charRNN.to(self.device)
        self.charRNN.eval()
        generations = []
        for i in range(int(amount)):
            generation = []
            charCount = 0
            print(f"Generation {i + 1}/{int(amount)}", end='\r')
            with torch.no_grad():
                hidden = self.charRNN.init_hidden(1, self.device)
                while True:
                    while current_n_gram.dim() < 4:
                        current_n_gram = current_n_gram.unsqueeze(0)
                    current_n_gram = current_n_gram.to(self.device)
                    logits, hidden = self.charRNN(current_n_gram, hidden)
                    next_idx = self.top_p_filtering(logits)
                    next_vec = torch.zeros(self.vocab_size, device=current_n_gram.device)
                    next_vec[next_idx] = 1
                    next_vec = next_vec.view(1, 1, -1)
                    char = self.endecode.decode(next_vec.squeeze(0))
                    charCount += 1
                    if char == '[EOS]' or charCount >= 400:
                        break
                    generation.append(char)
                    current_n_gram = next_vec.to(self.device)
            generations.append(''.join(generation))
        with open(filepath, 'w') as file:
            for item in generations:
                file.write(f"{item}\n")