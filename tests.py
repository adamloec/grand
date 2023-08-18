from grand.tensor import Tensor
from grand.layers import Layer, Input, Dense
from grand.nn import Model

# t1 = Tensor([1, 2, 3])
# t2 = Tensor([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])


# model = Model([
#     Input(2, 3),
#     Dense(4),
#     Dense(1)
#     ])

#model.forward()

import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

X, y = spiral_data(samples=100, classes=3)

print(X.shape)