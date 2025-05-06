import torch
from rdkit import Chem
from rdkit import RDLogger
from onehotencoder import OneHotEncoder
import numpy as np
from multiprocessing import Pool

class LossTerm:
    def __init__(self, logits, fail_sanitize=0.5, too_many_duplicates=0.2, wrong_value=0.1, fail_swaps=0.1):
        self.sanitize_cache = {}
        self.fail_sanitize = fail_sanitize
        self.too_many_duplicates = too_many_duplicates
        self.wrong_value = wrong_value
        self.fail_swaps = fail_swaps
        self.characters = ['Br', 'N', ')', 'c', 'o', '6', 's', 'Cl', '=', '2', ']', 'C', 'n', 'O', '4', '1', '#', 'S', 'F', '3', '[', '5', 'H', '(', '-', '[BOS]', '[EOS]', '[UNK]', '[PAD]']

        # Disable all RDKit warnings
        RDLogger.DisableLog('rdApp.*')

        self.endecode = OneHotEncoder()
        if isinstance(logits, torch.Tensor):
            self.logits = logits.detach().to('cpu')
        else:
            raise TypeError("logits must be a PyTorch tensor")

    # This function is used to convert the logits to SMILES strings
    def convert_logits(self):
        strings = []
        # Convert the result to one-hot encoded vectors then decode them to SMILES strings
        for i in range(self.logits.shape[0]):
            strings.append(self.endecode.decode_sequence(self.logits[i, :, :]))
        return strings

    # This function is used to check for common errors in SMILES strings
    def validation(self, string):
        # Initialize Variables
        val_loss = 0
        prev_char = None
        duplicate_count = 0
        RDLogger.DisableLog('rdApp.*')
        # Check for invalid start and end characters - missing [BOS] or [EOS]
        if string[:5] != '[BOS]':
            val_loss += self.wrong_value
        if string[-5:] != '[EOS]':
            val_loss += self.wrong_value
        # Check for invalid characters throughout the string
        for i in range(len(string)):
            # Check for invalid characters - UNK or PAD
            if string[i:i+5] in ('[UNK]', '[PAD]'):
                val_loss += self.wrong_value
            # Check for characters in invalid spots - [BOS] or [EOS]
            if i > 5 and string[i:i+5] == '[BOS]':
                val_loss += self.wrong_value
            if i < len(string) - 5 and string[i:i+5] == '[EOS]':
                val_loss += self.wrong_value
            
            # Check for too many duplicates
            if prev_char == string[i]:
                duplicate_count += 1
            else:
                val_loss += duplicate_count * self.too_many_duplicates
                duplicate_count = 0
            prev_char = string[i]
        return val_loss

    # This function uses the cache to get the results of the sanitize function or the original sanitize logic
    def sanitize(self, string):
        # Check if the string is already in the cache
        if string in self.sanitize_cache:
            return self.sanitize_cache.get(string) # return cached result if available
        RDLogger.DisableLog('rdApp.*')
        # If not in cache, sanitize the string and store the result
        result = self._sanitize(string) # original sanitize logic
        self.sanitize_cache.update({string: result})
        return result
    
    # This function is used to check if the SMILE string validates
    def _sanitize(self, string):
        RDLogger.DisableLog('rdApp.*')
        try:
            mol = Chem.MolFromSmiles(string, sanitize=True)
            if mol:
                return 1 # valid
            else:
                return 0 # invalid
        except Exception as e:
            print(f"Error sanitizing molecule: {e}")
            return 0 # invalid with error

    # This function is used to check if the string can becime valid by swapping characters
    def swap_loss(self, string):
        substitution_chars = ['c', 'C', '[', ']', '(', ')', '=', '', 'Br', 'N', 's', 'Cl', '6', '4', '2', '1', '5', '#', 'O', 'F', 'H', '3', 'S', '-']
        string_list = list(string)
        
        # Loop through the string and check for invalid characters
        for i in range(len(string)):
            # Save the original character
            original_char = string_list[i]
            
            # Try to swap with each substitution character
            for sub_char in substitution_chars:
                # Swap the character at index i with a substitution character
                string_list[i] = sub_char
                test_string = ''.join(string_list)
                
                # Check if the modified string is valid
                if self.sanitize(test_string) == 1:
                    return -((len(string) - i) * self.fail_swaps)
                
                # Revert the character to its original value
                string_list[i] = original_char
        # If no valid swap was found, return the number of swaps
        return len(string) * self.fail_swaps # return the number of swaps

    # This class is used to calculate the weights of the generated SMILES strings
    def evaluate_string(self, string):
        # Initialize the loss
        loss = 0
        
        # Check if it is an invalid molecule
        if self.sanitize(string) == 0:
            # Add the loss for being an invalid molecule
            loss += self.fail_sanitize
            # Add the loss for having invalid characters or too many duplicate characters
            loss += self.validation(string)
            # Add the loss for the number of swaps
            loss += self.swap_loss(string)
        
        # Return the loss as a positive value
        return max(loss, 0)

    # This function is used to calculate the loss for the batch
    def losses(self):
        # Get the sequences from the logits
        sequences = self.convert_logits()
        
        # Use multiprocessing to evaluate the strings
        with Pool() as pool:
            losses = pool.map(self.evaluate_string, sequences)
        
        # Calculate the average loss for the batch
        return np.log(np.mean(losses))
    