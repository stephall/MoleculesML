# smiles_one_hot_encoder.py

import numpy as np
from rdkit import Chem

class Encoder(object):
    # Default smiles characters containing all possible characters.
    _SMILES_CHARS = [
            ' ', '#', '%', '(', ')', '+', '-', '.', '/',
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            '=', '@',
            'A', 'B', 'C', 'F', 'H', 'I', 'K', 'L', 'M', 'N', 
            'O', 'P', 'R', 'S', 'T', 'V', 'X', 'Z',
            '[', '\\', ']',
            'a', 'b', 'c', 'e', 'g', 'i', 'l', 'n', 'o', 'p', 
            'r', 's', 't', 'u'
        ]

    def __init__(self, max_string_length, smiles_chars=None):
        """ Initialize the object. """
        # Assign inputs to class attributes
        self._max_string_length = max_string_length
        # In case smiles_char is not passed, use the full set
        if smiles_chars is None:
            self._smiles_chars  = self._SMILES_CHARS
        else:
            self._smiles_chars  = smiles_chars

        # Determine some attributes based on SMILES_CHARS
        self._num_chars  = len(self._smiles_chars)
        self._char2index = {char: index for index, char in enumerate(self._smiles_chars)}
        self._index2char = {index: char for index, char in enumerate(self._smiles_chars)}

    @property
    def dim_encoding(self):
        return self._max_string_length*self._num_chars

    def encode(self, smiles_string, flatten=True):
        """
        One-hot encode a smiles input string.
        
        Args:
            smiles_string (string): Smiles string.
            
        Output:
            (2d numpy array): Smiles one hot encoded matrix.
        """
        # Bring the smiles string into the expected fom using Chem from RDKit
        smiles_string = Chem.MolToSmiles( Chem.MolFromSmiles(smiles_string) )
        
        # Initialize the one hot matrix of shape (self._max_string_len, self._num_chars)
        smiles_one_hot = np.zeros( (self._max_string_length, self._num_chars) )
        
        # Loop over the characters of the smiles string and map them to
        # their corresponding one-hot-encoded vector composing the rows
        # of the smiles_one_hot matrix
        for index, char in enumerate(smiles_string):
            smiles_one_hot[index, self._char2index[char]] = 1

        if flatten:
            return smiles_one_hot.reshape(-1)
        else:
            return smiles_one_hot
     
    def decode(self, smiles_one_hot):
        """
        Decode a smiles one-hot input matrix to a smiles string.
        
        Args:
            smiles_one_hot (2d numpy array): Smiles one hot encoded matrix.
            
        Output:
            (string): Smiles string.
        """
        # Differ cases for smiles_one_hot
        if smiles_one_hot.ndim==1:
            # Check that the length is as expected
            expected_length = self._max_string_length*self._num_chars
            if len(smiles_one_hot)!=expected_length:
                err_msg = f"Input 'smiles_one_hot' is a 1d array of length {len(smiles_one_hot)} but the expected length is {expected_length}."
                raise ValueError(err_msg)

            # Reshape smiles_one_hot
            smiles_one_hot = smiles_one_hot.reshape( (self._max_string_length, self._num_chars) )
        elif smiles_one_hot.ndim==2:
            # Check that the shape is as expected
            expected_shape = (self._max_string_length, self._num_chars)
            if smiles_one_hot.shape!=expected_shape:
                err_msg = f"Input 'smiles_one_hot' is a 2d array of shape {smiles_one_hot.shape} but the expected shape is {expected_shape}."
                raise ValueError(err_msg)
        else:
            err_msg = f"Input 'smiles_one_hot' must be a 1d or 2d array, got dimension {smiles_one_hot.dim}."
            raise ValueError(err_msg)

        # initialize the smiles string as empty string
        smiles_string = ''
        
        # Make sure that smiles_one_hot is a float entried matrix
        smiles_one_hot.astype(float)
        
        # Loop over the rows of the one-hot smiles matrix 
        # Remark: These correspond to the one-hot encoded chars of the smiles string
        for row_index, row in enumerate(smiles_one_hot):
            # Check that the row is a one-hot encoded vector, meaning that
            # there is only one entry with value 1, while all the others are 0.
            # Remark: There are also rows with only 0 entries that corresponding to 
            #         empty characters for smiles strings that do not a number of
            #         max_string_length characters.
            if len(set(np.unique(row))-{0.0, 1.0})>0:
                print(row)
                err_msg = f"Row {row_index} of the one-hot encoded matrix contains elements different to 0 and 1."
                raise ValueError(err_msg)
            
            # Find the index that corresponds to 1, which is also 
            # the biggest entry in the row
            index = np.argmax(row)
            smiles_string += self._index2char[index]
            
        # Bring the smiles string into the expected fom using Chem from RDKit
        smiles_string = Chem.MolToSmiles( Chem.MolFromSmiles(smiles_string) )
            
        return smiles_string