from tensor import Tensor
from activations import Activation

class Dense:
    def __init__(self, size, activation=Activation.ReLU):
        self.size = size
        self.activation = activation