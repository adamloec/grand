from grand.tensor import Tensor
from grand.layers import Layer

from grand.utils import Colors

class Model:
    def __init__(self, layers=[]):
        self.layers = layers
        for layer in layers:
            if not isinstance(layer, Layer):
                raise TypeError("ERROR: Model inputs must all be layers")

        layers[0]._build(0, is_start=True)
        for i in range(1, len(layers)):
            layers[i]._build(layers[i-1].shape)

        self.loss = None
        self.optimizer = None

    def train(self, data, data_labels, batch_size=32, epochs=10):
        # Perform batching on both data/data_labels
        # Loop total number of epochs
        # Compare output to data_labels
        def __forward(data, data_labels):
            pass
        pass
    

    def view(self):
        print("\n--------------------------------------------------------")
        print(f"{Colors.HEADER}{Colors.UNDERLINE}Model View{Colors.ENDC}\n")
        for i, layer in enumerate(self.layers):
            print(f"{Colors.OKGREEN}Layer: {Colors.ENDC}{i}, Type: '{type(layer).__name__}', Shape: {layer.shape}, Activation: {layer.activation.__name__}\n")
        print("--------------------------------------------------------\n")