import numpy as np
from grand.tensor import Tensor

class Activation:

    def ReLU(x):
        if not isinstance(x, Tensor):
            raise TypeError("ERROR: Data B must be of type Tensor")
        return Tensor(np.maximum(0.0, x.data))
    
    def Linear(x):
        if not isinstance(x, Tensor):
            raise TypeError("ERROR: Data B must be of type Tensor")
        return x

    # def sigmoid(x):
    #     if not isinstance(x, (int, float)):
    #         raise TypeError("ERROR: Data B must be of type int, float")
    #     return 1/(1+np.exp(-x))