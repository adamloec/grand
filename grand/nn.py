from tensor import Tensor
from activations import Activation

class Dense:
    def __init__(self, size, data=None, activation=Activation.ReLU):
        self.size = size
        self.activation = activation
        self.data = data

        # create weights and biases, random tensor creation dependent on size of dense layer, weights will be of size (X input, Y output) or Dense(input, neuron)
        # create forward method that calculates logistic regression (z = wx + b), z is output dependent on number of neurons. Ex: Dense(2 input, 3 neuron)
        # apply activation function on output values, all values equal to neuron #
        # create backward method that calculates gradient