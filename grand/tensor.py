# tensor.py
# Author: Adam
# Description: Tensor object and utilities. Built on numpy, utilizing Cuda kernels for GPU operations.
# 
#
#

import numpy as np

class Tensor:
    def __init__(self, data, dtype=np.float32):
        """
        Name: Tensor() \n
        Purpose: Parent constructor for the Tensor object. \n
        Parameters: data (list, np.ndarray), dtype (Any) \n
        Return: None \n

        Uses: \n
        t = Tensor([1, 2, 3]) \n
        t = Tensor(np.array([1, 2, 3])) \n
        """

        if not isinstance(data, (np.ndarray, list)):
            raise TypeError("ERROR: Data must be of type list, np.ndarray")
        data = np.array(data, dtype=dtype)
        self.data = data
        self.shape = self.data.shape
        self.dtype = dtype

    @classmethod
    def zeros(cls, *shape, dtype=np.float32):
        """
        Name: Tensor.zeros() \n
        Purpose: Class method for creating Tensor object with zeros, given shape. \n
        Parameters: *shape (N number of integers), dtype (Any) \n
        Return: Tensor \n

        Uses: \n
        t = Tensor.zeros(1, 2) \n
        """

        for d in shape:
            if not isinstance(d, int):
                raise TypeError("ERROR: shape must be of type int")
        t = cls(data=np.zeros(shape, dtype=dtype), dtype=dtype)
        return t
    
    @classmethod
    def ones(cls, *shape, dtype=np.float32):
        """
        Name: Tensor.ones() \n
        Purpose: Class method for creating Tensor object with ones, given shape. \n
        Parameters: *shape (N number of integers), dtype (Any) \n
        Return: Tensor \n

        Uses: \n
        t = Tensor.ones(1, 2) \n
        """

        for d in shape:
            if not isinstance(d, int):
                raise TypeError("ERROR: shape must be of type int")
        t = cls(data=np.ones(shape, dtype=dtype), dtype=dtype)
        return t
    
    @classmethod
    def rand(cls, *shape, dtype=np.float32):
        """
        Name: Tensor.rand() \n
        Purpose: Class method for creating Tensor object with random values, given shape. \n
        Parameters: *shape (N number of integers), dtype (Any) \n
        Return: Tensor \n

        Uses: \n
        t = Tensor.rand(1, 2) \n
        """

        for d in shape:
            if not isinstance(d, int):
                raise TypeError("ERROR: shape must be of type int")
        t = cls(data=np.random.uniform(0, 1, shape), dtype=dtype)
        return t
    
    @classmethod
    def empty(cls, *shape, dtype=np.float32):
        """
        Name: Tensor.empty() \n
        Purpose: Class method for creating an empty Tensor object, given shape. \n
        Parameters: *shape (N number of integers), dtype (Any) \n
        Return: Tensor \n

        Uses: \n
        t = Tensor.empty(1, 2) \n
        """

        for d in shape:
            if not isinstance(d, int):
                raise TypeError("ERROR: shape must be of type int")
        t = cls(data=np.empty(shape, dtype=dtype), dtype=dtype)
        return t
    
    def __add__(self, b):
        """
        Name: __add__() \n
        Purpose: Addition function, between 2 Tensors or Tensor and integer/float \n
        Parameters: b (Tensor, integer, float) \n
        Return: Tensor \n

        Uses: \n
        Tensor = Tensor + Tensor \n
        Tensor = Tensor + integer/float \n
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
        Name: __mul__() \n
        Purpose: Multiplication function, between 2 Tensors or Tensor and integer/float \n
        Parameters: b (Tensor, integer, float) \n
        Return: Tensor \n

        Uses: \n
        Tensor = Tensor * Tensor \n
        Tensor = Tensor * integer/float \n
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
        Name: __matmul__() \n
        Purpose: Matrix Multiplication function, between 2 Tensors \n
        Parameters: b (Tensor) \n
        Return: Tensor \n

        Uses: \n
        Tensor = Tensor @ Tensor \n

        Requirements: \n
        n*k @ k*m = n*m matrix \n
        """

        if isinstance(b, Tensor):
            #assert self.shape == b.shape, "ERROR: A and B input Tensors must be the same size"
            return Tensor(np.matmul(self.data, b.data))
        else:
            raise TypeError("ERROR: Data B must be of type Tensor, int, float")

    def __str__(self):
        """
        Name: __str__() \n
        Purpose: Str function, used for printing Tensor contents \n
        Parameters: None \n
        Return: String \n

        Uses: \n
        print(Tensor) \n
        """

        return str(self.data)