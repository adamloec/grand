from tensor import Tensor
from ops import Ops

class Dense:
    def __init__(self, size, activation=Ops.ReLU):
        self.size = size
        self.activation = activation