from grand.tensor import Tensor
from grand.layers import Layer, Input

class Model:
    def __init__(self, layers=[]):
        self.layers = layers
        for i, layer in enumerate(self.layers):
            if not isinstance(layer, Input):
                layer.build(self.layers[i-1])

    def view(self):
        for i, layer in enumerate(self.layers):
            print(f"Layer: {i}, Type: {type(layer)}")
            print("Weights:\n", layer.weights)
            print("Biases:\n", layer.biases)

    def forward(self):
        for i, layer in enumerate(self.layers):
            if not isinstance(layer, Input):
                layer.forward(self.layers[i-1].output)