# tensor.py
# Author: Adam
# Description: Tensor object and utilities. Built on numpy, utilizing Cuda kernels for GPU operations.
# 
#
#

import numpy as np

class Tensor:
    """
    Name: Tensor()
    Purpose: Parent constructor for the Tensor object.
    Parameters: data (list, np.ndarray), dtype (Any)
    Return: None

    Uses: \n
    t = Tensor([1, 2, 3])
    t = Tensor(np.array([1, 2, 3]))
    """
    def __init__(self, data, dtype=np.float32, device='gpu'):

        if not isinstance(data, (np.ndarray, list)):
            raise TypeError("ERROR: Data must be of type list, np.ndarray")
        self.data = np.array(data, dtype=dtype)
        self.shape = self.data.shape
        self.dtype = dtype

    @classmethod
    def zeros(cls, *shape, dtype=np.float32):
        """
        Name: Tensor.zeros()
        Purpose: Class method for creating Tensor object with zeros, given shape.
        Parameters: *shape (N number of integers), dtype (Any)
        Return: Tensor

        Uses: \n
        t = Tensor.zeros(1, 2)
        """

        for d in shape:
            if not isinstance(d, int):
                raise TypeError("ERROR: shape must be of type int")
        t = cls(data=np.zeros(shape, dtype=dtype), dtype=dtype)
        return t
    
    @classmethod
    def ones(cls, *shape, dtype=np.float32):
        """
        Name: Tensor.ones()
        Purpose: Class method for creating Tensor object with ones, given shape.
        Parameters: *shape (N number of integers), dtype (Any)
        Return: Tensor

        Uses: \n
        t = Tensor.ones(1, 2)
        """

        for d in shape:
            if not isinstance(d, int):
                raise TypeError("ERROR: shape must be of type int")
        t = cls(data=np.ones(shape, dtype=dtype), dtype=dtype)
        return t
    
    @classmethod
    def rand(cls, *shape, dtype=np.float32, seed=0):
        """
        Name: Tensor.rand()
        Purpose: Class method for creating Tensor object with random values, given shape.
        Parameters: *shape (N number of integers), dtype (Any)
        Return: Tensor

        Uses: \n
        t = Tensor.rand(1, 2)
        """

        for d in shape:
            if not isinstance(d, int):
                raise TypeError("ERROR: shape must be of type int")
        np.random.seed(seed)
        t = cls(data=np.random.uniform(0, 1, shape), dtype=dtype)
        return t
    
    @classmethod
    def empty(cls, *shape, dtype=np.float32):
        """
        Name: Tensor.empty()
        Purpose: Class method for creating an empty Tensor object, given shape.
        Parameters: *shape (N number of integers), dtype (Any)
        Return: Tensor

        Uses: \n
        t = Tensor.empty(1, 2)
        """

        for d in shape:
            if not isinstance(d, int):
                raise TypeError("ERROR: shape must be of type int")
        t = cls(data=np.empty(shape, dtype=dtype), dtype=dtype)
        return t
    
    def __add__(self, b):
        """
        Name: __add__()
        Purpose: Addition function, between 2 Tensors or Tensor and integer/float.
        Parameters: b (Tensor, integer, float)
        Return: Tensor

        Uses: \n
        Tensor = Tensor + Tensor
        Tensor = Tensor + integer/float
        """

        if isinstance(b, Tensor):
            assert self.shape == b.shape, "ERROR: A and B input Tensors must be the same size"
            return Tensor(self.data + b.data)
        elif isinstance(b, (int, float)):
            return Tensor(self.data + b)
        else:
            raise TypeError("ERROR: Data B must be of type Tensor, int, float")
        
    def __radd__(self, b):
        """
        Name: __radd__()
        Purpose: Addition function, between integer/float and Tensor.
        Parameters: b (Tensor, integer, float)
        Return: Tensor

        Uses: \n
        Tensor = integer/float + Tensor
        """

        if isinstance(b, Tensor):
            assert self.shape == b.shape, "ERROR: A and B input Tensors must be the same size"
            return Tensor(self.data + b.data)
        elif isinstance(b, (int, float)):
            return Tensor(self.data + b)
        else:
            raise TypeError("ERROR: Data B must be of type Tensor, int, float")
            
    def __mul__(self, b):
        """
        Name: __mul__()
        Purpose: Multiplication function, between 2 Tensors or Tensor and integer/float.
        Parameters: b (Tensor, integer, float)
        Return: Tensor

        Uses: \n
        Tensor = Tensor * Tensor
        Tensor = Tensor * integer/float
        """

        if isinstance(b, Tensor):
            assert self.shape == b.shape, "ERROR: A and B input Tensors must be the same size"
            return Tensor(self.data * b.data)
        elif isinstance(b, (int, float)):
            return Tensor(self.data * b)
        else:
            raise TypeError("ERROR: Data B must be of type Tensor, int, float")
        
    def __rmul__(self, b):
        """
        Name: __rmul__()
        Purpose: Multiplication function, between integer/float and Tensor.
        Parameters: b (Tensor, integer, float)
        Return: Tensor

        Uses: \n
        Tensor = integer/float * Tensor
        """

        if isinstance(b, Tensor):
            assert self.shape == b.shape, "ERROR: A and B input Tensors must be the same size"
            return Tensor(self.data * b.data)
        elif isinstance(b, (int, float)):
            return Tensor(self.data * b)
        else:
            raise TypeError("ERROR: Data B must be of type Tensor, int, float")
        
    def __matmul__(self, b):
        """
        Name: __matmul__()
        Purpose: Matrix Multiplication function, between 2 Tensors.
        Parameters: b (Tensor)
        Return: Tensor

        Uses: \n
        Tensor = Tensor @ Tensor

        Requirements: \n
        n*k @ k*m = n*m matrix
        """

        if isinstance(b, Tensor):
            #assert self.shape == b.shape, "ERROR: A and B input Tensors must be the same size"
            return Tensor(np.matmul(self.data, b.data))
        else:
            raise TypeError("ERROR: Data B must be of type Tensor, int, float")

    def __str__(self):
        """
        Name: __str__()
        Purpose: Str function, used for printing Tensor contents.
        Parameters: None
        Return: String

        Uses: \n
        print(Tensor)
        """

        return str(self.data)
    
    def reshape(self, *shape):
        reshaped = self.data.reshape(shape)
        return Tensor(reshaped, self.dtype)