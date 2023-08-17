import numpy as np
from grand.tensor import Tensor
from grand.activations import Activation

# Parent Layer object
class Layer:
    def __init__(self, *dim, dtype=np.float32):
        self.dtype = dtype
        self.output = Tensor.empty(*dim, dtype=self.dtype)
        self.dim = self.output.dim

        self.weights = None
        self.biases = None
    
    def __str__(self):
        return f'\n Type: {type(self).__name__} \n Dims: {self.dim} \n {self.data.__str__()}'

# Input layer object
class Input(Layer):
    def __init__(self, *dim, dtype=np.float32):
        super().__init__(*dim, dtype=dtype)

    def __call__(self, input):        
        if not isinstance(input, (Tensor, list, np.ndarray)):
            raise TypeError("ERROR: Data must be of type Tensor, list, np.ndarray")
        if isinstance(input, (list, np.ndarray)):
            self.output = Tensor(input)

        if self.dim != self.output.dim:
            raise Exception("ERROR: Input data dimensions do not match input layer")

        return self.output
    
    # Forward pass for input layer
    # NOTE: Will always be a linear activation, no w/b in input layer.
    def forward(self, input):
        if not isinstance(input, Tensor):
            raise TypeError("ERROR: Input must be of type Tensor")
        self.output = input
        return self.output

# Dense layer object
class Dense(Layer):
    def __init__(self, size, *, dtype=np.float32, activation=Activation.ReLU):
        if not isinstance(size, int):
            raise TypeError("ERROR: Dense layer size must be single integer")
        
        super().__init__(size, dtype=dtype)
        self.activation = activation

    def build(self, layer):
        self.weights = Tensor.rand(layer.dim[-1], self.dim[0])
        self.biases = Tensor.rand(self.dim[0])

    # Forward pass for dense layer
    def forward(self, input):
        if not isinstance(input, Tensor):
            raise TypeError("ERROR: Input must be of type Tensor")
        if self.weights == None or self.biases == None:
            raise Exception("ERROR: Weights and biases have not been created")
        
        self.output = input @ self.weights + self.biases
        # Apply activation function before returning output
        return self.output