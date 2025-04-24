import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from rdkit import Chem
from rdkit.Chem import SanitizeFlags
from rdkit.Chem import MolToSmiles
from onehotencoder import onehotencoder

class lossTerm():
    def __init__(self, logits, batch_size, fail_sanitize, too_many_duplicates, wrong_value, missing_end_bracket, fail_swaps):
        self.batch_size = batch_size
        self.fail_sanitize = fail_sanitize
        self.too_many_duplicates = too_many_duplicates
        self.wrong_value = wrong_value
        self.missing_end_bracket = missing_end_bracket
        self.fail_swaps = fail_swaps
        self.characters = ['Br', 'N', ')', 'c', 'o', '6', 's', 'Cl', '=', '2', ']', 'C', 'n', 'O', '4', '1', '#', 'S', 'F', '3', '[', '5', 'H', '(', '-', '[BOS]', '[EOS]', '[UNK]', '[PAD]']

        self.endecode = onehotencoder()
        if isinstance(logits, torch.Tensor):
            self.logits = logits.to('cpu').detach().numpy()
        else:
            raise TypeError("logits must be a PyTorch tensor")

    # Convert the result to one-hot encoded vectors then decode them to SMILES strings
    # This function is used to convert the logits to SMILES strings
    def convert_logits(self):
        strings = []
        for i in range(self.logits.size(dim=2)):
            for j in range(self.logits.size(dim=1)):
                strings.append(self.endecode.decode_sequence(self.logits[i,j,:]))
        return strings

    # This function is used to check the validity of the generated SMILES strings
    def validation(self, string):
        loss = 0
        n = len(string)
        i = 0
        for i in range(n):
            # Check for invalid characters
            if string[i:i+5] == '[UNK]':
                loss += self.wrong_value
            if string[i:i+5] == '[PAD]':
                loss += self.wrong_value
            if string[0:5] != '[BOS]':
                loss += self.wrong_value
            if string[n-5:n] != '[EOS]':
                loss += self.wrong_value
            # Check for too many duplicate characters
            if (string[i-5:i-1] == 'cccccccccc' and string[i] == 'c'):
                loss += self.too_many_duplicates
            if (string[i-5:i-1] == 'CCCCCCCCCC' and string[i] == 'C'):
                loss += self.too_many_duplicates
            if (string[i-13:i-1] == 'ClClClClClCl' and string[i:i+1] == 'Cl'):
                loss += self.too_many_duplicates
            if (string[i-5:i-1] == '[Br]' and string[i:i+4] == '[Br]'):
                loss += self.too_many_duplicates
            if ((string[i-1] == '1' and string[i] == '1') or (string[i-1] == '2' and string[i] == '2') or (string[i-1] == '3' and string[i] == '3') or (string[i-1] == '4' and string[i] == '4') or (string[i-1] == '5' and string[i] == '5') or (string[i-1] == '6' and string[i] == '6')):
                loss += self.too_many_duplicates
        return loss

    # The first check to give loss if the string is not valid
    def sanitize_check(self, string):
        try:
            mol = Chem.MolFromSmiles(string)
            if mol is None:
                return self.fail_sanitize
            else:
                Chem.SanitizeMol(mol, sanitizeFlags=SanitizeFlags.SANITIZE_ALL)
                return 0
        except Exception as e:
            print(f"Error sanitizing molecule: {e}")
            return self.fail_sanitize

    
    # This function is used to check if the SMILE string will validate - used after swaps
    def sanitize(self, string):
        try:
            mol = Chem.MolFromSmiles(string)
            if mol is None:
                return 1
            Chem.SanitizeMol(mol, sanitizeFlags=SanitizeFlags.SANITIZE_ALL)
            return 0
        except Exception as e:
            print(f"Error sanitizing molecule: {e}")
            return 1
    
    # This function is used to check the swap loss of the generated SMILES strings
    def change_loss(self, char_list, int_current):
        string = ''.join(char_list)
        integer = 0

        # If it is fixed subtract the number of swaps from the integer
        if (self.sanitize(string) == 0):
            int_current -= 1
            integer -= int_current # subtract the number of swaps
        else:
            int_current += 1
            integer += int_current # add the number of swaps
        return integer

    # This function is used to check the swap loss of the generated SMILES strings
    def swap_loss(self, string):
        stringList = list(string)
        string1, string2, string3, string4, string5, string6, string7, string8 = stringList.copy(), stringList.copy(), stringList.copy(), stringList.copy(), stringList.copy(), stringList.copy(), stringList.copy(), stringList.copy()
        integer, int1, int2, int3, int4, int5, int6, int7, int8 = 0, 0, 0, 0, 0, 0, 0, 0, 0
        for i in range(len(string)):
            # if the fix occurs stop the loop and give the negative value as the return
            string1[i] = 'c' # check c
            integer = self.change_loss(string1[i], int1)
            if (integer < 0):
                return integer
            
            string2[i] = 'C' # check C
            integer = self.change_loss(string2[i], int2)
            if (integer < 0):
                return integer
                
            string3[i] = '[' # open []
            integer = self.change_loss(string3[i], int3)
            if (integer < 0):
                return integer

            string4[i] = ']' # close []
            integer = self.change_loss(string4[i], int4)
            if (integer < 0):
                return integer

            string5[i] = '(' # open ()
            integer = self.change_loss(string5[i], int5)
            if (integer < 0):
                return integer

            string6[i] = ')' # close ()
            integer = self.change_loss(string6[i], int6)
            if (integer < 0):
                return integer

            string7[i] = '=' # bond7
            integer = self.change_loss(string7[i], int7)
            if (integer < 0):
                return integer

            string8[i] = '' # reduce the string by one char
            integer = self.change_loss(string8[i], int8)
            if (integer < 0):
                return integer
        return integer

    # This class is used to calculate the weights of the generated SMILES strings
    def losses(self):
        losses = []
        loss = 0
        lossTerm = 0

        # Convert the logits to SMILES strings
        sequences = self.convert_logits(self.logits)

        # Iterate through the sequences
        # For each string, calculate the weights
        for string in sequences:
            # Add the loss for having invalid characters or too many duplicate characters
            validate = self.validation(string)
            loss += validate
            # Add the loss for being an invalid molecule
            sanitize = self.sanitize_check(string)
            loss += sanitize
            # Add the loss for the number of swaps
            loss += self.swap_loss(string)
            # Check if the loss is negative
            if (loss < 0):
                loss = 0
            
            # Save the loss for this sequence/string
            losses.append(loss)
            loss = 0
        
        # Calculate the average weight
        for i in range(len(losses)):
            lossTerm += losses[i]
        lossTerm = lossTerm / len(losses)

        # Print and return the average weight
        print(f"Average weight: {lossTerm}")
        return lossTerm