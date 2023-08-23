import numpy as np
from grand.tensor import Tensor
from grand.activations import Activation

class Layer:
    def __init__(self, units=0, *, input_shape=0, dtype=np.float32, activation=Activation.Linear):
        if input_shape != 0:
            if not isinstance(input_shape, tuple):
                raise TypeError("ERROR: input_shape must be of type tuple")
            self.shape = (None,) + tuple(input_shape,)
        else: 
            self.shape = (None, units)

        self.dtype = dtype
        self.activation = activation
        
        self.weights = None
        self.biases = None
        self.output = None

class Dense(Layer):
    def __init__(self, units, *, input_shape=0, dtype=np.float32, activation=Activation.ReLU):
        super(Dense, self).__init__(units, input_shape=input_shape, dtype=dtype, activation=activation)

    def _build(self, prev_shape):
        if not isinstance(prev_shape, tuple):
                raise TypeError("ERROR: input_shape must be of type tuple")
        self.weights = Tensor.rand(prev_shape[-1], self.shape[-1])
        self.biases = Tensor.rand(1, self.shape[-1])

    def forward(self, data):
        if not isinstance(input, Tensor):
            raise TypeError("ERROR: Input must be of type Tensor")
        if self.weights == None or self.biases == None:
            raise Exception("ERROR: Weights and biases have not been created")
        self.output = (data @ self.weights) + self.biases
        return self.output # Activation function here

# Will always be input layer, flattens shape of input data
class Flatten(Layer):
    def __init__(self, input_shape=0, dtype=np.float32):
        if not isinstance(input_shape, tuple):
                raise TypeError("ERROR: input_shape must be of type tuple")
        reshaped = 1
        for s in input_shape:
            reshaped *=s
        super(Flatten, self).__init__(0, input_shape=(reshaped,), dtype=dtype)

    def forward(self, data):
        if not isinstance(data, Tensor):
            raise TypeError("ERROR: Input data must be of type Tensor")
        return data