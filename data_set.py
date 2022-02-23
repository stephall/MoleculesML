#data_set.py

# Import some modules
import os
import numpy as np
import pandas as pd
import smiles_one_hot_encoder
import torch
import transformation
from torch.utils.data import Dataset
from imp import reload

# Reload some modules
reload(smiles_one_hot_encoder)
reload(transformation)

class CustomDataset(Dataset):
    def __init__(self, data_dir, x_encoding_dict, y_label, data_filter_dict=None, transformations_x=None, transformations_y=None):
        # Assign inputs to class attributes
        self.data_dir          = data_dir
        self.y_label           = y_label
        self._x_encoding_dict  = x_encoding_dict
        self._data_filter_dict = data_filter_dict

        # Set the precision of the torch tensors
        self._floating_point_precision = np.float32

        # Load the data as pandas DataFrame
        self.data_df = self.load_data()

        # Initalize an encoder to None (not relevant for all x encodings)
        self._x_encoder = None

        # Generate the encoded x
        self._x_encoded = self.generate_encoded_x()

        # Try to find the index of the column that corresponds to the y_label,
        # and throw an error if it is not found
        try:
            self._col_index_y_label = list(self.data_df.columns).index(self.y_label)
        except ValueError:
            err_msg = f"The input 'y_label' was {self.y_label} but must correspond to one " \
                      f"of the following:\n{list(self.data_df.columns)}"
            raise ValueError(err_msg)

        # Initialize the transformations
        self._transformer_x = transformation.Transformer(transformations_x, self.get_x(transform=False))
        self._transformer_y = transformation.Transformer(transformations_y, self.get_y(transform=False))

        print()

    def __len__(self):
        """ Return the number of data item. """
        return len(self.data_df)

    def __getitem__(self, index):
        """
        Return one data item corresponding to the passed index.

        Args:
            index (int): Index of the to be returned data item.

        Returns:
            (tuple): x and y value of the data item
        """
        # Get the x value corresponding to the index
        x = self._x_encoded[index]

        # Get the y value corresponding to the index
        y = np.array( [self.data_df.iloc[index, self._col_index_y_label]] ).astype(self._floating_point_precision)

        # Apply the transformations to both x and y.
        x = self.transform_x(x)
        y = self.transform_y(y)

        # Map the x and y to the correct floating point precision
        x = x.astype(self._floating_point_precision)
        y = y.astype(self._floating_point_precision)

        return x, y

    def transform_x(self, x):
        """
        Transform x using the corresponding transformer.
        """
        return self._transformer_x.transform(x)

    def transform_y(self, y):
        """
        Transform y using the corresponding transformer.
        """
        return self._transformer_y.transform(y)

    def inverse_transform_x(self, x):
        """
        Inverse transform x using the corresponding transformer.
        """
        return self._transformer_x.inverse_transform(x)

    def inverse_transform_y(self, y):
        """
        Inverse transform y using the corresponding transformer.
        """
        return self._transformer_y.inverse_transform(y)

    @property
    def y(self):
        # Return the y values of the data
        return self.get_y(transform=True)

    def get_y(self, transform=True):
        # Return the y values of the data
        y = np.array(self.data_df[self.y_label]).reshape(-1, 1)

        # In case that transform is False, transform the data
        if transform==True:
            y = self.transform_y(y)

        return y

    @property
    def x(self):
        # Return the x values of the data
        return self.get_x(transform=True)

    def get_x(self, transform=True):
        # Return the x values of the data
        x = self._x_encoded

        # In case that transform is False, transform the data
        if transform==True:
            x = self.transform_x(x)

        return x

    @property
    def x_dim(self):
        # Differ cases
        if self._x_encoding_dict['which']=='one_hot':
            return self._x_encoder.dim_encoding
        elif self._x_encoding_dict['which']=='finger_prints':
            raise NotImplementedError('Not implemented')
        else:
            err_msg = f"The 'x_encoding' must be 'one_hot' or 'finger_prints'."
            raise ValueError(err_msg)

    @property
    def y_dim(self):
        return self.y.shape[1]
    

    def load_data(self):
        """
        Load the data from the data directory and filter it if rquested.

        Args:
            None

        Returns:
            (pandas.DataFrame): Loaded (and maybe filtered) data as pandas data frame.
        """
        # Check if the data directory exists
        if not os.path.isfile(self.data_dir):
            err_msg = f"The input 'data_dir' is not valid, got the input: \n{self.data_dir}"
            raise FileNotFoundError(err_msg)

        # Load the DataFrame from pickle file
        data_df = pd.read_pickle(self.data_dir)
        print(f"Loaded data from '{self.data_dir}'.")

        # Filter the data if requested
        if self._data_filter_dict is not None:
            data_df = self.filter_data(data_df)

        return data_df

    def filter_data(self, data_df):
        """
        Filter the data.
        
        Args:
            data_df (pandas.DataFrame): Data to be filtered.

        Returns: 
            (pandas.DataFrame): Filtered data.
        """
        # Differ cases
        if self._data_filter_dict['which']=='max_string_length_cutoff':
            # Store the unfiltered data size
            unfiltered_data_len = len(data_df)

            # Filter by a maximal smiles string length
            mask    = data_df['smiles'].str.len() <= self._data_filter_dict['params']['max_string_length']
            data_df = data_df.loc[mask]

            # Display what has been done
            msg = f"Filtered the data using '{self._data_filter_dict['which']}' method.\nReduction from " \
                  f"{unfiltered_data_len} to {len(data_df)} data items."
            print(msg)

        else:
            err_msg = f"Can not filter by '{self._data_filter_dict['which']}', not implemented."
            raise NotImplementedError(err_msg)

        return data_df

    def generate_encoded_x(self):
        """
        Generate the encoded x array.

        Args:
            None

        Returns:
            (2d numpy array): Encoded x array of shape (#data_items, encoding_dimension)
        """
        print('Generate encoding of x ...')
        # Differ cases
        if self._x_encoding_dict['which']=='one_hot':
            self.initialize_one_hot_x_encoder()

        # Loop over the smiles strings and encode each of them to a 1d array
        smiles_encoded_list = list()
        for smiles_string in self.data_df['smiles']:
            # Encode the smiles string
            # Remark: This 1d array will have shape (encoding_dimension,)
            smiles_encoded = self.encode_x(smiles_string)

            # Append the encoded smiles string
            smiles_encoded_list.append(smiles_encoded)

        # Stack the encoded smiles to 2d array of shape (#data_items, encoding_dimension)
        # and return it
        smiles_encoded_array = np.vstack(smiles_encoded_list)

        print('Encoding done.')

        return smiles_encoded_array

    def initialize_one_hot_x_encoder(self):
        """
        Initialize the x encoder, which is not relevant for all x encodings.
        """
        # Loop over the smiles and create a set of all their characters
        char_set = set()
        for smiles_string in self.data_df['smiles']:
            # Loop over the characters in the smiles_string and add them to the char set
            for char in smiles_string:
                char_set.add(char)

        # Initialize the x_encoder as one hot encoder 
        self._x_encoder = smiles_one_hot_encoder.Encoder(self._x_encoding_dict['params']['max_string_length'], list(char_set))

    def encode_x(self, x_string):
        """
        Encode the x value.

        Args:
            x_string (str): To be encoded x (smiles string)

        Returns:
            (1d numpy array): Encoded x value.
        """
        # Differ cases
        if self._x_encoding_dict['which']=='one_hot':
            return self._x_encoder.encode(x_string)
        elif self._x_encoding_dict['which']=='finger_prints':
            raise NotImplementedError('Not implemented')
        else:
            err_msg = f"The 'x_encoding' must be 'one_hot' or 'finger_prints'."
            raise ValueError(err_msg)

    def decode_x(self, x_encoded):
        """
        Decode the encoded x value.

        Args:
            x_encoded (1d numpy array): To be decoded x value.

        Returns:
            (str): Decoded x value (smiles string).
        """
        # Differ cases
        if self._x_encoding_dict['which']=='one_hot':
            return self._x_encoder.decode(x_encoded)
        elif self._x_encoding_dict['which']=='finger_prints':
            raise NotImplementedError('Not implemented')
        else:
            err_msg = f"The 'x_encoding' must be 'one_hot' or 'finger_prints'."
            raise ValueError(err_msg)

