import numpy as np
from grand.tensor import Tensor
from grand.activations import Activation

from grand.utils import Colors

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

        self._is_input = False
        self._has_built = False

    def _build(self, prev_shape=(0,), is_input=False):
        if is_input:
            self._is_input = True
            return
        if not isinstance(prev_shape, tuple):
                raise TypeError("ERROR: input_shape must be of type tuple")
        
        self.weights = Tensor.rand(prev_shape[-1], self.shape[-1])
        self.biases = Tensor.rand(1, self.shape[-1])
        self._has_built = True

    def view(self):
        print("\n--------------------------------------------------------")
        if self._has_built and not self._is_input:
            print(f"{Colors.HEADER}{Colors.UNDERLINE}Layer View{Colors.ENDC}\n")
            print(f"{Colors.OKGREEN}Weights: {Colors.ENDC}Shape: {self.weights.shape}\n")
            print(f"{Colors.OKGREEN}Biases: {Colors.ENDC}Shape: {self.biases.shape}\n")
        else:
            print("ERROR: Layer is input and/or compile model to view weights/biases.")

class Dense(Layer):
    def __init__(self, units, *, input_shape=0, dtype=np.float32, activation=Activation.Linear):
        super(Dense, self).__init__(units, input_shape=input_shape, dtype=dtype, activation=activation)

    def forward(self, data):
        if not isinstance(input, Tensor):
            raise TypeError("ERROR: Input must be of type Tensor")
        if self.weights == None or self.biases == None:
            raise Exception("ERROR: Weights and biases have not been created")
        
        self.output = (data @ self.weights) + self.biases
        return self.activation(self.output)

# Will always be a 'filler' layer, will not perform forward/backprop func
class Flatten(Layer):
    def __init__(self, input_shape=0, dtype=np.float32):
        if not isinstance(input_shape, tuple):
                raise TypeError("ERROR: input_shape must be of type tuple")
        
        self.reshaped = 1
        for s in input_shape:
            self.reshaped *= s
        super(Flatten, self).__init__(0, input_shape=(self.reshaped,), dtype=dtype)

    def forward(self, data):
        if not isinstance(data, Tensor):
            raise TypeError("ERROR: Input data must be of type Tensor")
        
        self.output = data.reshape(data[0], self.reshaped)
        return self.activation(self.output)