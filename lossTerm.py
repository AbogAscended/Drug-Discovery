import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from rdkit import Chem
from rdkit.Chem import SanitizeFlags
from rdkit.Chem import MolToSmiles
from onehotencoder import onehotencoder

class lossTerm():
    def __init__(self, logits, batch_size, fail_sanitize, missing_string_boundary, wrong_value_in_padding, no_pad_token, fail_swaps):
        self.batch_size = batch_size
        self.fail_sanitize = fail_sanitize
        self.missing_string_boundary = missing_string_boundary
        self.wrong_value_in_padding = wrong_value_in_padding
        self.no_pad_token = no_pad_token
        self.fail_swaps = fail_swaps

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
        # First Check: Starts with BOS token
        if string[0:4] != '[BOS]':
            loss += self.missing_string_boundary
        # Second Check: EOS token is present 
        eos_index = string.find('[EOS]')
        if eos_index == -1:
            loss += self.missing_string_boundary
            #Third Check: check for a PAD token
            first_pad_index = string.find('[PAD]')
            if first_pad_index != -1:
                weight += self.no_pad_token
            else:
                # Third Check: Only PAD tokens are present anywhere # TODO
                pad_index = string.find('[EOS]')
                if pad_index != -1:
                    loss += self.wrong_value_in_padding
                # Fourth Check: Check for the length of the string
                if n > 59:
                    loss += self.wrong_value_in_padding
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
            int_current += 1
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
        strings = self.convert_logits(self.logits)

        # Iterate through the generated SMILES strings
        for string in strings:
            # For each string, calculate the weights
            validate = self.validation(string)
            loss += validate
            sanitize = self.sanitize_check(string)
            loss += sanitize
            loss += self.swap_loss(string)
            losses.append(loss)
            loss = 0
        
        # Calculate the average weight
        for i in range(len(losses)):
            lossTerm += losses[i]
        lossTerm = lossTerm / len(losses)

        print(f"Average weight: {lossTerm}")
        return lossTerm