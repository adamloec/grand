from grand.tensor import Tensor

class Loss:
    def __init__(self):
        pass

class CategoricalCrossentropy(Loss):

    def _forward(self, prediction, true):
        if not isinstance(prediction, Tensor):
            raise TypeError("ERROR: Loss data type must be of type Tensor")
        if not isinstance(true, Tensor):
            raise TypeError("ERROR: Loss data type must be of type Tensor")  