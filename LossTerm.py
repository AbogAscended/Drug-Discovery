import torch
from rdkit import Chem
from rdkit.Chem import SanitizeFlags
from onehotencoder import OneHotEncoder

class LossTerm():
    def __init__(self, logits, batch_size, fail_sanitize, too_many_duplicates, wrong_value, fail_swaps):
        self.batch_size = batch_size
        self.fail_sanitize = fail_sanitize
        self.too_many_duplicates = too_many_duplicates
        self.wrong_value = wrong_value
        self.fail_swaps = fail_swaps
        self.characters = ['Br', 'N', ')', 'c', 'o', '6', 's', 'Cl', '=', '2', ']', 'C', 'n', 'O', '4', '1', '#', 'S', 'F', '3', '[', '5', 'H', '(', '-', '[BOS]', '[EOS]', '[UNK]', '[PAD]']

        self.endecode = OneHotEncoder()
        if isinstance(logits, torch.Tensor):
            self.logits = logits.to('cpu').detach().numpy()
        else:
            raise TypeError("logits must be a PyTorch tensor")

    # Convert the result to one-hot encoded vectors then decode them to SMILES strings
    # This function is used to convert the logits to SMILES strings
    def convert_logits(self):
        strings = []
        for i in range(self.logits.shape[0]):
            for j in range(self.logits.shape[1]):
                strings.append(self.endecode.decode_sequence(self.logits[i, j, :]))
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
            if string[i - 5:i - 1] == 'cccccccccc' and string[i] == 'c':
                loss += self.too_many_duplicates
            if string[i - 5:i - 1] == 'CCCCCCCCCC' and string[i] == 'C':
                loss += self.too_many_duplicates
            if string[i - 13:i - 1] == 'ClClClClClCl' and string[i:i + 1] == 'Cl':
                loss += self.too_many_duplicates
            if string[i - 5:i - 1] == '[Br]' and string[i:i + 4] == '[Br]':
                loss += self.too_many_duplicates
            if (string[i - 1] == '1' and string[i] == '1') or (string[i - 1] == '2' and string[i] == '2') or (string[i - 1] == '3' and string[i] == '3') or (string[i - 1] == '4' and string[i] == '4') or (string[i - 1] == '5' and string[i] == '5') or (string[i - 1] == '6' and string[i] == '6'):
                loss += self.too_many_duplicates
        print("Validation: ", loss)
        return loss

    # The first check to give loss if the string is not valid
    def sanitize_check(self, string):
        try:
            mol = Chem.MolFromSmiles(string)
            if mol is None:
                print("Sanitization failed: Invalid SMILES string")
                return self.fail_sanitize
            else:
                Chem.SanitizeMol(mol, sanitizeOps=SanitizeFlags.SANITIZE_ALL)
                print("Sanitization successful")
                return 0
        except Exception as e:
            print(f"Error sanitizing molecule: {e}")
            return self.fail_sanitize

    
    # This function is used to check if the SMILE string will validate - used after swaps
    def sanitize(self, string):
        try:
            mol = Chem.MolFromSmiles(string)
            if mol is None:
                print("Swap Sanitization failed")
                return 1
            Chem.SanitizeMol(mol, sanitizeOps=SanitizeFlags.SANITIZE_ALL)
            print("Swaps successful")
            return 0
        except Exception as e:
            print(f"Error sanitizing molecule: {e}")
            return 1

    # This function is used to check the swap loss of the generated SMILES strings
    def swap_loss(self, string):
        string_list = list(string)
        string1, string2, string3, string4, string5, string6, string7, string8 = string_list.copy(), string_list.copy(), string_list.copy(), string_list.copy(), string_list.copy(), string_list.copy(), string_list.copy(), string_list.copy()
        integer = 0
        for i in range(len(string)):
            # if the fix occurs stop the loop and give the negative value as the return
            string1[i] = 'c' # check c
            string_c = ''.join(string1)
            if self.sanitize(string_c) == 0:
                integer += i # subtract the number of swaps
                return -(integer * self.fail_swaps)
            
            string2[i] = 'C' # check C
            string_C = ''.join(string2)
            if self.sanitize(string_C) == 0:
                integer += i # subtract the number of swaps
                return -(integer * self.fail_swaps)
                
            string3[i] = '[' # open []
            string_open_b = ''.join(string3)
            if self.sanitize(string_open_b) == 0:
                integer += i # subtract the number of swaps
                return -(integer * self.fail_swaps)

            string4[i] = ']' # close []
            string_close_b = ''.join(string4)
            if self.sanitize(string_close_b) == 0:
                integer += i # subtract the number of swaps
                return -(integer * self.fail_swaps)

            string5[i] = '(' # open ()
            string_openpr = ''.join(string5)
            if self.sanitize(string_openpr) == 0:
                integer += i # subtract the number of swaps
                return -(integer * self.fail_swaps)

            string6[i] = ')' # close ()
            string_close = ''.join(string6)
            if self.sanitize(string_close) == 0:
                integer += i # subtract the number of swaps
                return -(integer * self.fail_swaps)

            string7[i] = '=' # bond7
            string_equ = ''.join(string7)
            if self.sanitize(string_equ) == 0:
                integer += i # subtract the number of swaps
                return -(integer * self.fail_swaps)

            string8[i] = '' # reduce the string by one char
            string_null = ''.join(string8)
            if self.sanitize(string_null) == 0:
                integer += i # subtract the number of swaps
                return -(integer * self.fail_swaps)
        print("Swaps: ", len(string) * self.fail_swaps)
        print("Swap weight: ", (len(string) * self.fail_swaps))
        return len(string) * self.fail_swaps # return the number of swaps

    # This class is used to calculate the weights of the generated SMILES strings
    def losses(self):
        losses = []
        loss = 0
        loss_term = 0

        # Convert the logits to SMILES strings
        sequences = self.convert_logits()

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
            
            # Save the loss for this sequence/string
            losses.append(loss)
            loss = 0
        
        # Calculate the average weight
        for i in range(len(losses)):
            loss_term += losses[i]
        loss_term = loss_term / len(losses)

        # Print and return the average weight
        print(f"Average weight: {loss_term}")
        return loss_term