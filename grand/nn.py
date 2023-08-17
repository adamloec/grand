import numpy as np
from grand.tensor import Tensor
from grand.activations import Activation

class Layer:
    def __init__(self, *dim, dtype=np.float32):
        self.data = Tensor.zeros(*dim, dtype=dtype)
        self.dim = self.data.dim

    def __call__(self):
        print(self.__str__())
    
    def __str__(self):
        return f'\n Type: {type(self).__name__} \n Dims: {self.dim} \n {self.data.__str__()}'

class Input(Layer):
    def __init__(self, *dim, dtype=np.float32):
        super().__init__(*dim, dtype=dtype)

    # Should be __call__ (Forward) method, should verify input data matches dimensions of Input layer
    # Only layer where type conversion will happen from list/np.ndarray -> Tensor
    def forward(self, input):
        if not isinstance(input, (Tensor, list, np.ndarray)):
            raise TypeError("ERROR: Data must be of type Tensor, list, np.ndarray")
        if isinstance(input, (list, np.ndarray)):
            input = Tensor(input)
        
        return input

class Dense(Layer):
    def __init__(self, size, *, dtype=np.float32, activation=Activation.ReLU):
        if not isinstance(size, int):
            raise TypeError("ERROR: Dense layer size must be single integer")
        
        super().__init__(size, dtype=dtype)
        self.activation = activation
        self.weights = None # dim = (input*size)
        self.biases = None # dim = (size)

    # Forward pass, initializes random weights and biases on first call, checks for dimensions of input and self.dim
    def __call__(self, input=None):
        if input == None:
            super(Dense, self).__call__()
            return
        if not isinstance(input, Tensor):
            raise TypeError("ERROR: Input must be of type Tensor")
        
        if self.weights and self.biases == None:
            self.weights = Tensor.rand()
        print(input)

# class Dense:
#     def __init__(self, size, dtype=np.float32, activation=Activation.ReLU):

#         self.size = size # Int, size of output neurons
#         self.activation = activation
#         self.dtype = dtype

#         self.w = None # Weights
#         self.b = None # Biases
#         self.output = None # Activation values a_hat

#     def init_params(self, indim):
#         self.w = Tensor.rand(indim, self.size)
#         self.b = Tensor.rand(self.size)
    
#     def forward(self, inputs):
#         # regression, activation function
#         pass

#         # create weights and biases, random tensor creation dependent on size of dense layer, weights will be of size (X input, Y output) or Dense(input, neuron)
#         # create forward method that calculates logistic regression (z = wx + b), z is output dependent on number of neurons. Ex: Dense(2 input, 3 neuron)
#         # apply activation function on output values, all values equal to neuron #
#         # create backward method that calculates gradient

#         # Making assumption, at the moment, that all models will be sequential