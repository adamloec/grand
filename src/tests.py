from tensor import Tensor

t1 = Tensor([1, 2, 3])
t2 = Tensor([1, 2, 3])
t3 = t1 @ t2
print(t3)