#transformation.py

import numpy as np

class Transformer(object):
    # Define the allowed transformations
    _allowed_operations = ['+', '-', '*', '/']

    def __init__(self, transformations, data):
        """
        Initialize a transformation object.

        Args:
            transformations (list or str): Either a list of transformations 
                in form [(operation, scalar/array),...], a string equivalent to 
                'standardize' as well 'normalize' or None.
                For 'standardize' or 'normalize', the transformation will be created.
                For None, no transformation will be applied.

            data (numpy.array): A 2d array corresponding to the to be transformed 
                data.
                This is actually only needed if transformations='standardization'
                and not used otherwise.

        """
        # In case that transformations are 'standardize' or 'normalize', generate
        # the transformations corresponding to a standardization
        # of the input data.
        if transformations==None:
            transformations = []
        elif transformations=='standardize':
            transformations = self._generate_standardization_transformations(data)
        elif transformations=='normalize':
            transformations = self._generate_normalization_transformations(data)
        else:
            # Check that transformations is a list
            if not isinstance(transformations, list):
                err_msg = f"The input 'transformations' must be 'standardize', 'normalize' or a list, but " \
                          f"the following was passed:\n {transformations}"
                raise TypeError(err_msg)

        # Check the transformations
        self._check_transformations(transformations)

        # Assign the preprocessed and checked input transformations to class attribute
        self._transformations = transformations

    def _generate_standardization_transformations(self, data):
        """
        Generate the transformations (list) corresponding to a standardization of the data.
        This way the data is transformed to mean 0 and std 1.

        Args:
            data (numpy.array): Numpy array of shape (N,M).

        Returns:
            (list): Transformations corresponding to a standardization of the data along
                the first axis.

        """
        # Check that the data is a 2d array
        if not isinstance(data, np.ndarray):
            err_msg = f"The input 'data' must be a 2d numpy array, got type '{type(data)}' instead."
            return TypeError(err_msg)

        if data.ndim!=2:
            err_msg = f"The input 'data' must be a 2d numpy array, got dimension {data.ndim} instead."
            return ValueError(err_msg)

        # Obtain the mean and standard deviation along the first axis of the data
        mean_data = np.mean(data, axis=0)
        std_data  = np.std(data, axis=0)

        # Generate the standardization operation x->(x-mean)/std
        standardization_transformations = [('-', mean_data), ('/', std_data)]

        return standardization_transformations

    def _generate_normalization_transformations(self, data):
        """
        Generate the transformations (list) corresponding to a normalization of the data.
        This way the data is transformed to [0, 1]

        Args:
            data (numpy.array): Numpy array of shape (N,M).

        Returns:
            (list): Transformations corresponding to a normalization of the data along
                the first axis.

        """
        # Check that the data is a 2d array
        if not isinstance(data, np.ndarray):
            err_msg = f"The input 'data' must be a 2d numpy array, got type '{type(data)}' instead."
            return TypeError(err_msg)

        if data.ndim!=2:
            err_msg = f"The input 'data' must be a 2d numpy array, got dimension {data.ndim} instead."
            return ValueError(err_msg)

        # Obtain the minimum and maximum along the first axis of the data
        min_data = np.min(data, axis=0)
        max_data  = np.max(data, axis=0)

        # Generate the normalization operation x->(x-x_min)/(x_max-x_min)
        normalization_transformations = [('-', min_data), ('/', (max_data-min_data))]

        return normalization_transformations

    def _check_transformations(self, transformations):
        """
        Check that the transformations correspond to the expected format.

        Args:
            transformations (list): To be checked list of transformations that 
                should contain 2-tuples of the form (operation, scalar/array)
        """
        # Loop over the transformations and check them
        for transformation in transformations:
            # Check that the transformation is a tuple
            if not isinstance(transformation, tuple):
                err_msg = f"A single transformation must be a tuple, got type '{type(transformation)}' instead."
                raise TypeError(err_msg)

            # Check that the first element of the tuple 'transformation' (operation)
            # is one of the allowed operations
            if transformation[0] not in self._allowed_operations: 
                err_msg = f"The first element of a transformation (operation) must be one of the following:\n {self._allowed_operations}\n" \
                          f"But got operation '{transformation[0]}' instead."
                raise ValueError(err_msg)

            # Check that the second element of the tuple 'transformation' is a scalar or numpy array
            if not isinstance(transformation[1], (float, int, np.ndarray)): 
                err_msg = f"The second element of a transformation must of type 'float', 'int', or 'np.ndarray', got type '{type(transformation[1])}' instead."
                raise TypeError(err_msg)

            # Check that if the operation (first element of the transformation) is '*' or '/',
            # the corresponding scalar/array (second element of the transformation) is not 0
            # for a scalar and does not contain 0 elements in case of array
            if transformation[0] in ['*', '/']:
                if isinstance(transformation[1], (float, int)):
                    if transformation[1]==0:
                        err_msg = f"In case that the operation (first element of a transformation) is '*' or '/', the " \
                                  f"corresponding scalar (second element of a transformation) can not be zero!\n" \
                                  f"Got the transformation: {transformation}"
                        raise ValueError(err_msg)
                
                    if isinstance(transformation[1], np.ndarray):
                        if np.any(transformation[1]==0):
                            err_msg = f"In case that the operation (first element of a transformation) is '*' or '/', the " \
                                      f"corresponding array (second element of a transformation) can not contain any 0!\n" \
                                      f"Got the transformation: {transformation}"
                            raise ValueError(err_msg)

    def transform(self, x):
        """
        Transform an input x.

        Args:
            x (numpy.array): The to be transformed input.
        Returns:
            (numpy.array): The transformed input.

        """
        # Loop over the transformations and apply them
        for transformation in self._transformations:
            x = self._transform_operation(x, transformation[0], transformation[1])

        return x

    def inverse_transform(self, x):
        """
        Inverse the transformation of a transformed x.

        Args:
            x (numpy.array): The to be inversely transformed input.
        Returns:
            (numpy.array): The inversely transformed input.

        """
        # Loop over the transformations in revesed order and apply the
        # inverse transformation for each of them
        for transformation in self._transformations[::-1]:
            x = self._inverse_transform_operation(x, transformation[0], transformation[1])

        return x

    def _transform_operation(self, x, operation, s):
        """
        Execute a (binary) operation '$' for two inputs x and s.

        Args:
            x (numpy.array): Array of shape (N, M).
            operation (str): Operation '$' to be applied to x and s.
            s (float, int, numpy.array): Scalar or array.

        Returns:
            (numpy.array): Result of the operation 'x$s', which is
                and array of shape (N,M)

        """
        # Differ cases for the operations
        if operation=='+':
            return x+s
        elif operation=='-':
            return x-s
        elif operation=='*':
            return x*s
        elif operation=='/':
            return x/s
        else:
            err_msg = f"Operation must be '+','-', '*', or '/', got '{operation}' instead."
            raise ValueError(err_msg)

    def _inverse_transform_operation(self, x, operation, s):
        """
        Execute an inverse (binary) operation to operation '$' for two inputs x and s.

        Args:
            x (numpy.array): Array of shape (N, M).
            operation (str): Inverse operation to operation '$' to be applied to x and s.
            s (float, int, numpy.array): Scalar or array.

        Returns:
            (numpy.array): Result of the inverse operation to 'x$s', 
                which is and array of shape (N,M)

        """
        # Differ cases for the operations
        if operation=='+':
            return x-s
        elif operation=='-':
            return x+s
        elif operation=='*':
            return x/s
        elif operation=='/':
            return x*s
        else:
            err_msg = f"Operation must be '+','-', '*', or '/', got '{operation}' instead."
            raise ValueError(err_msg)
          




