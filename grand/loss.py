from grand.tensor import Tensor


class CategoricalCrossentropy:

    def _forward(self, prediction, true):
        if not isinstance(prediction, Tensor):
            raise TypeError("ERROR: Loss data type must be of type Tensor")
        if not isinstance(true, Tensor):
            raise TypeError("ERROR: Loss data type must be of type Tensor")
        
        samples = len(prediction.data)

        