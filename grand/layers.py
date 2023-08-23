import numpy as np
from grand.tensor import Tensor
from grand.activations import Activation

# Dense layer object
class Dense:
    def __init__(self, size, *, dtype=np.float32, activation=Activation.ReLU):
        if not isinstance(size, int):
            raise TypeError("ERROR: Dense layer size must be single integer")
        
        self.size = size
        self.dtype = dtype
        self.activation = activation

        self.output = None
        self.weights = None
        self.biases = None
    
    def __str__(self):
        return f'\n Type: {type(self).__name__} \n Shape: {self.shape} \n {self.data.__str__()}'

    def _build(self, layer):
        self.weights = Tensor.rand(layer.size, self.size)
        self.biases = Tensor.rand(1, self.size)

    # Forward pass for dense layer
    def forward(self, input):
        if not isinstance(input, Tensor):
            raise TypeError("ERROR: Input must be of type Tensor")
        if self.weights == None or self.biases == None:
            raise Exception("ERROR: Weights and biases have not been created")

        self.output = (input @ self.weights) + self.biases
        return self.activation(self.output)
    
# Psuedo model
# 
# Dense(10)
# Dense(10, 5)
# Dense(5, 2) -> 2 output neurons, multi-classification

# input data 5
# dense(10)
#