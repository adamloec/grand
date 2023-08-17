import numpy as np

class Tensor:
    def __init__(self, data, dtype=np.float32):
        if not isinstance(data, (np.ndarray, list)):
            raise TypeError("ERROR: Data must be of type list, np.ndarray")
        data = np.array(data, dtype=dtype)
        self.data = data
        self.dim = data.shape

    @classmethod
    def zeros(cls, *dim, dtype=np.float32):
        for d in dim:
            if not isinstance(d, int):
                raise TypeError("ERROR: Dimensions must be of type int")
        t = cls(data=np.zeros(dim), dtype=dtype)
        return t
    
    @classmethod
    def ones(cls, *dim, dtype=np.float32):
        for d in dim:
            if not isinstance(d, int):
                raise TypeError("ERROR: Dimensions must be of type int")
        t = cls(data=np.ones(dim), dtype=dtype)
        return t
    
    @classmethod
    def rand(cls, *dim, dtype=np.float32):
        for d in dim:
            if not isinstance(d, int):
                raise TypeError("ERROR: Dimensions must be of type int")
        t = cls(data=np.random.randn(dim))
        return t
    
    def __add__(self, b):
        if isinstance(b, Tensor):
            assert self.dim == b.dim, "ERROR: A and B input Tensors must be the same size"
            return Tensor(self.data + b.data)
        elif isinstance(b, (int, float)):
            return Tensor(self.data + b)
        else:
            raise TypeError("ERROR: Data B must be of type Tensor, int, float")
            
    def __mul__(self, b):
        if isinstance(b, Tensor):
            assert self.dim == b.dim, "ERROR: A and B input Tensors must be the same size"
            return Tensor(self.data * b.data)
        elif isinstance(b, (int, float)):
            return Tensor(self.data * b)
        else:
            raise TypeError("ERROR: Data B must be of type Tensor, int, float")
        
    def __matmul__(self, b):
        if isinstance(b, Tensor):
            assert self.dim == b.dim, "ERROR: A and B input Tensors must be the same size"
            return np.matmul(self.data, b.data)
        else:
            raise TypeError("ERROR: Data B must be of type Tensor, int, float")
        
    def __call__(self):
        print(self.data)

    def __str__(self):
        return str(self.data)