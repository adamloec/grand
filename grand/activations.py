import numpy as np

class Activation:

    def ReLU(x):
        if not isinstance(x, (int, float)):
            raise TypeError("ERROR: Data B must be of type int, float")
        return np.max(0.0, x)

    def sigmoid(x):
        if not isinstance(x, (int, float)):
            raise TypeError("ERROR: Data B must be of type int, float")
        return 1/(1+np.exp(-x))