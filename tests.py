from grand.tensor import Tensor
from grand.nn import Layer, Input, Dense

t1 = Tensor([1, 2, 3])
t2 = Tensor([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])

inp = Input(2, 3)
dense = Dense(3)
dense()