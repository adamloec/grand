from grand.tensor import Tensor
from grand.layers import Layer, Input, Dense

t1 = Tensor([1, 2, 3])
t2 = Tensor([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])

inp = Input(2, 3)
dense = Dense(4)

dense(inp())

class Model:
    def __init__(self, layers=[]):
        self.layers = layers

        

    # Compile layers dimensions to match for training, errors if not matching
    def build(self):
        pass


    def forward(self):
        for layer in self.layers:
            pass