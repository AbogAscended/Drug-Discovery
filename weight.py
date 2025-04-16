import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from rdkit import Chem
from rdkit.Chem import SanitizeFlags
from rdkit.Chem import MolToSmiles
from onehotencoder import onehotencoder

batch_size=29
fail_sanitize=1
missing_string_boundary=5
wrong_value_in_padding=2
no_pad_token=2
fail_sanitize = 1
fail_swaps = 5

# Convert the result to one-hot encoded vectors then decode them to SMILES strings
# This function is used to convert the logits to SMILES strings
def convert_logits(logits):
    endecode = onehotencoder()
    encoded = endecode.encode_sequence(logits)
    strings = endecode.decode_sequence(encoded)
    return strings

# This function is used to check the validity of the generated SMILES strings
def validation(string, missing_string_boundary, wrong_value_in_padding, no_pad_token):
    weight = 0
    n = len(string)
    # First Check: Starts with BOS token
    if string[0:4] != '[BOS]':
        weight += missing_string_boundary
    # Second Check: EOS token is present 
    eos_index = string.find('[EOS]')
    if eos_index == -1:
        weight += missing_string_boundary
        #Third Check: check for a PAD token
        first_pad_index = string.find('[PAD]')
        if first_pad_index != -1:
            weight += no_pad_token
        else:
            # Third Check: Only PAD tokens are present after EOS token
            for i in range((first_pad_index+5), n):
                if string[i:i+4] != '[PAD]':
                    weight += wrong_value_in_padding
                i += 5
    else:
        # Third Check: Only PAD tokens are present after EOS token
        for j in range((eos_index+5), n):
            if string[j:j+4] != '[PAD]':
                weight += wrong_value_in_padding
            j += 5
    return weight

# This function is used to check the validity of the generated SMILES strings Check and add a weight
def sanitize_check(string, fail_sanitize):
    try:
        mol = Chem.MolFromSmiles(string)
        if mol is None:
            return fail_sanitize
        else:
            Chem.SanitizeMol(mol, sanitizeFlags=SanitizeFlags.SANITIZE_ALL)
            return 0
    except Exception as e:
        print(f"Error sanitizing molecule: {e}")
        return fail_sanitize
    
# This function is used to check if the SMILE string will validate
def sanitize(string):
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
def change_loss(char_list, int_current):
    string = ''.join(char_list)
    integer = 0

    # If it is fixed subtract the number of swaps from the integer
    if (sanitize(string) == 0):
        int_current += 1 # add one more
        integer -= int_current # subtract the number of swaps
    else:
        int_current += 1 # add one more
        integer += int_current # add the number of swaps
    return integer

# This function is used to check the swap loss of the generated SMILES strings
def swap_loss(string):
    stringList = list(string)
    string1, string2, string3, string4, string5, string6, string7, string8 = stringList.copy(), stringList.copy(), stringList.copy(), stringList.copy(), stringList.copy(), stringList.copy(), stringList.copy(), stringList.copy()
    integer, int1, int2, int3, int4, int5, int6, int7, int8 = 0, 0, 0, 0, 0, 0, 0, 0, 0
    for i in range(len(string)):
        # if the fix occurs stop the loop and give the negative value as the return
        string1[i] = 'c' # check c
        int1 = i
        integer = change_loss(string1[i], int1)
        # check for fixed molecule
        if (integer < 0):
            return integer
        
        string2[i] = 'C' # check C
        int2 = i
        integer = change_loss(string2[i], int2)
        # check for fixed molecule
        if (integer < 0):
            return integer
            
        string3[i] = '[' # open []
        int3 = i
        integer = change_loss(string3[i], int3)
        # check for fixed molecule
        if (integer < 0):
            return integer

        string4[i] = ']' # close []
        int4 = i
        integer = change_loss(string4[i], int4)
        # check for fixed molecule
        if (integer < 0):
            return integer

        string5[i] = '(' # open ()7
        int5 = i
        integer = change_loss(string5[i], int5)
        # check for fixed molecule
        if (integer < 0):
            return integer

        string6[i] = ')' # close ()7
        int6 = i
        integer = change_loss(string6[i], int6)
        # check for fixed molecule
        if (integer < 0):
            return integer

        string7[i] = '=' # bond7
        int7 = i
        integer = change_loss(string7[i], int7)
        # check for fixed molecule
        if (integer < 0):
            return integer

        string8[i] = '' # reduce the string by one char
        int8 = i
        integer = change_loss(string8[i], int8)
        # check for fixed molecule
        if (integer < 0):
            return integer
    return integer

# This class is used to calculate the weights of the generated SMILES strings
def weights(logits, missing_string_boundary, wrong_value_in_padding, no_pad_token, fail_sanitize):
    weights = []
    weight = 0
    weightFinal = 0

    # Convert the logits to SMILES strings
    # strings = convert_logits(logits)
    strings = ['CC(=O)C1=CC=CC=C1C(=O)O', 'CC(=O)C1=CC=CC=C1C(=O)O', 'CC(=O)C1=CC=CC=C1C(=O)O'] # Example strings for testing

    # Iterate through the generated SMILES strings
    for string in strings:
        # For each string, calculate the weights
        validate = validation(string, missing_string_boundary, wrong_value_in_padding, no_pad_token)
        weight += validate
        sanitize = sanitize_check(string, fail_sanitize)
        weight += sanitize
        weight += swap_loss(string)
        weights.append(weight)
    
    # Calculate the average weight
    for i in range(len(weights)):
        weightFinal += weights[i]
    weightFinal = weightFinal / len(weights)

    print(f"Average weight: {weightFinal}")
    return weightFinal

weights(logits=None, missing_string_boundary=missing_string_boundary, wrong_value_in_padding=wrong_value_in_padding, no_pad_token=no_pad_token, fail_sanitize=fail_sanitize)