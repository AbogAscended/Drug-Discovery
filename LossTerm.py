import torch
from rdkit import Chem
from rdkit import RDLogger
from onehotencoder import OneHotEncoder
import numpy as np

class LossTerm:
    def __init__(self, logits, fail_sanitize=0.5, too_many_duplicates=0.2, wrong_value=0.1, fail_swaps=0.1):
        self.fail_sanitize = fail_sanitize
        self.too_many_duplicates = too_many_duplicates
        self.wrong_value = wrong_value
        self.fail_swaps = fail_swaps
        self.characters = ['Br', 'N', ')', 'c', 'o', '6', 's', 'Cl', '=', '2', ']', 'C', 'n', 'O', '4', '1', '#', 'S', 'F', '3', '[', '5', 'H', '(', '-', '[BOS]', '[EOS]', '[UNK]', '[PAD]']

        # Disable all RDKit warnings
        RDLogger.DisableLog('rdApp.*')

        self.endecode = OneHotEncoder()
        if isinstance(logits, torch.Tensor):
            self.logits = logits.to('cpu').detach()
        else:
            raise TypeError("logits must be a PyTorch tensor")

    # This function is used to convert the logits to SMILES strings
    def convert_logits(self):
        strings = []
        # Convert the result to one-hot encoded vectors then decode them to SMILES strings
        for i in range(self.logits.shape[0]):
            strings.append(self.endecode.decode_sequence(self.logits[i, :, :]))
        return strings

    # This function is used to check the validity of the generated SMILES strings
    def validation(self, string):
        loss = 0
        n = len(string)
        i = 0
        dictionary = dict.fromkeys(self.characters, 0)
        for i in range(n):
            # Check for invalid characters
            if string[i:i+5] == '[UNK]':
                loss += self.wrong_value
            if string[i:i+5] == '[PAD]':
                loss += self.wrong_value
            # Check for too many duplicate characters
            if string[i - 1] == string[i]:
                dictionary[string[i]] += self.too_many_duplicates
            if i > 5:
                if string[i:i+5] == '[BOS]':
                    loss += self.wrong_value
            if i < n - 5:
                if string[i:i+5] == '[EOS]':
                    loss += self.wrong_value
        if string[0:5] != '[BOS]':
            loss += self.wrong_value
        if string[n - 5:n] != '[EOS]':
            loss += self.wrong_value

        # final loss from validation
        loss += sum(dictionary.values())
        return loss

    # This function is used to check if the SMILE string validates
    def sanitize(self, string):
        try:
            mol = Chem.MolFromSmiles(string, sanitize=True)
            if mol:
                return 0 # valid
            else:
                return 1 # invalid
        except Exception as e:
            print(f"Error sanitizing molecule: {e}")
            return 1 # invalid with error

    # This function is used to check the swap loss of the generated SMILES strings
    def swap_loss(self, string):
        string_list = list(string)
        string1, string2, string3, string4 = string_list.copy(), string_list.copy(), string_list.copy(), string_list.copy()
        string5, string6, string7, string8 = string_list.copy(), string_list.copy(), string_list.copy(), string_list.copy()
        string9, string10, string11, string12 = string_list.copy(), string_list.copy(), string_list.copy(), string_list.copy()
        string13, string14, string15, string16 = string_list.copy(), string_list.copy(), string_list.copy(), string_list.copy()
        string17, string18, string19, string20 = string_list.copy(), string_list.copy(), string_list.copy(), string_list.copy()
        string21 , string22, string23, string24 = string_list.copy(), string_list.copy(), string_list.copy(), string_list.copy()

        # iterate through the string copies at the same time
        for i in range(len(string)):
            # change the char at i index in the string-char list
            string1[i], string2[i], string3[i], string4[i] = 'c', 'C', '[', ']' # check c # check C # open [ # close ]
            string5[i], string6[i], string7[i], string8[i] = '(', ')', '=', '' # open ( # close () # bond = # reduce ''
            string9[i], string10[i], string11[i], string12[i] = 'Br', 'N', 's', 'Cl' # check Br # check N # check s # check N
            string13[i], string14[i], string15[i], string16[i], string17[i] = '6', '4', '2', '1', '5' # check 6, 4, 2, 1, 5
            string18[i], string19[i], string20[i], string21[i] = '#', 'O', 'F', 'H' # bond '#' # check O # check F, H
            string22[i], string23[i], string24[i] = '3', 'S', '-' # check 3 # check S # check -

            # For each character make the list into a string
            string_c, string_upper_c, string_open_b, string_close_b = ''.join(string1), ''.join(string2), ''.join(string3), ''.join(string4)
            string_open_pr, string_close_pr, string_equ, string_null = ''.join(string5), ''.join(string6), ''.join(string7), ''.join(string8)
            string_br, string_n, string_s, string_cl  = ''.join(string9), ''.join(string10), ''.join(string11), ''.join(string12)
            string_six, string_four, string_two = ''.join(string13), ''.join(string14), ''.join(string15)
            string_one, string_five, string_hash = ''.join(string16), ''.join(string17), ''.join(string18)
            string_o, string_f, string_h = ''.join(string19), ''.join(string18), ''.join(string19)
            string_three, string_upper_s, string_hyphen = ''.join(string20), ''.join(string21), ''.join(string22)

            # save sanitize checks
            sanitize_checks = [self.sanitize(string_c), self.sanitize(string_upper_c), self.sanitize(string_open_b), self.sanitize(string_close_b),
                     self.sanitize(string_open_pr), self.sanitize(string_close_pr), self.sanitize(string_equ), self.sanitize(string_null),
                     self.sanitize(string_br), self.sanitize(string_n), self.sanitize(string_s), self.sanitize(string_cl),
                     self.sanitize(string_six), self.sanitize(string_four), self.sanitize(string_two), self.sanitize(string_one),
                     self.sanitize(string_five), self.sanitize(string_hash), self.sanitize(string_o), self.sanitize(string_f),
                     self.sanitize(string_h), self.sanitize(string_three), self.sanitize(string_upper_s), self.sanitize(string_hyphen)]

            # if the fix occurs, stop the loop and give the negative value as the return
            if any(sanitize_checks) == 0:
                integer = (len(string) - i) * self.fail_swaps
                return -(integer * self.fail_swaps)

        # Otherwise, swaps failed (so full penalty)
        return len(string) * self.fail_swaps # return the number of swaps

    # This class is used to calculate the weights of the generated SMILES strings
    def losses(self):
        losses = []
        loss, loss_term = 0, 0

        # Convert the logits to SMILES strings
        sequences = self.convert_logits()

        # Iterate through the sequences
        # For each string, calculate the weights
        for string in sequences:
            # Check if it is an invalid molecule
            if self.sanitize(string) == 1:
                # Add the loss for being an invalid molecule
                loss += self.fail_sanitize

                # Add the loss for having invalid characters or too many duplicate characters
                validate = self.validation(string)
                loss += validate

                # Add the loss for the number of swaps
                loss += self.swap_loss(string)
            else:
                loss = 0
            
            # Save the loss for this sequence/string
            if loss < 0:
                loss = 0
            losses.append(loss)
            loss = 0 # Clean loss for the next string

        # Calculate the average weight
        for i in range(len(losses)):
            loss_term += losses[i]
        loss_term = loss_term / len(losses)

        # Print and return the average weight
        # print(f"Average weight: {loss_term}")
        return np.log(loss_term)