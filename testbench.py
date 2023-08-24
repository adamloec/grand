import numpy as np
import random
import time

from grand.tensor import Tensor
from grand.layers import Layer, Dense, Flatten
from grand.nn import Model
from grand.activations import Activation

from grand.utils import Colors

class Testbench:
    """
    Name: Testbench()
    Purpose: Constructor for the Testbench object.
    Parameters: func (mathematical operation function), runs (int)
    Return: None

    Uses: \n
    Testbench(tensor_tensor_add)
    """
    def __init__(self, func, runs=1000):
        self.func = func
        self.runs = runs

        tic = time.perf_counter()
        for run in range(runs):

            dim1, dim2 = random.randint(1, 100), random.randint(1, 100)
            np1 = np.array(np.random.rand(dim1, dim2), dtype=np.float32)
            np2 = np.array(np.random.rand(dim1, dim2), dtype=np.float32)

            if not func(dim1=dim1, dim2=dim2, np1=np1, np2=np2):
                print(f"TEST {func.__name__} {run}/{runs} {Colors.FAIL}FAIL{Colors.ENDC}")
            else:
                if run % 100 == 0:
                    print(f"TEST {func.__name__} Runs: {run}/{runs} {Colors.OKGREEN}PASS{Colors.ENDC}")
        toc = time.perf_counter()
        
        print(f"TEST {func.__name__} Runs: {runs}/{runs} {Colors.OKGREEN}PASS{Colors.ENDC}") 
        print(f"Duration: {toc - tic:0.4f}s")

def tensor_tensor_add(**kwargs):
    """
    Name: tensor_tensor_add()
    Purpose: Test function for addition between 2 tensor's.
    Parameters: **kwargs(dim1, dim2, np1, np2)
    Return: Bool

    Uses: \n
    Intended to be used inside of Testbench(func)
    """
    tensor_result = np.around((Tensor(kwargs['np1']) + Tensor(kwargs['np2'])).data, decimals=4)
    np_result = np.around(kwargs['np1'] + kwargs['np2'], decimals=4)
    return (tensor_result == np_result).all()

def tensor_scalar_add(**kwargs):
    """
    Name: tensor_scalar_add()
    Purpose: Test function for addition between Tensor and Scalar (integer/float).
    Parameters: **kwargs(dim1, dim2, np1, np2)
    Return: Bool

    Uses: \n
    Intended to be used inside of Testbench(func)
    """
    tensor_result = np.around((Tensor(kwargs['np1']) + kwargs['dim1']).data, decimals=4)
    np_result = np.around(kwargs['np1'] + kwargs['dim1'], decimals=4)
    return (tensor_result == np_result).all()

def scalar_tensor_add(**kwargs):
    """
    Name: scalar_tensor_add()
    Purpose: Test function for addition between Scalar (integer/float) and Tensor.
    Parameters: **kwargs(dim1, dim2, np1, np2)
    Return: Bool

    Uses: \n
    Intended to be used inside of Testbench(func)
    """
    tensor_result = np.around((kwargs['dim1'] + Tensor(kwargs['np1'])).data, decimals=4)
    np_result = np.around(kwargs['dim1'] + kwargs['np1'], decimals=4)
    return (tensor_result == np_result).all()

def tensor_tensor_mul(**kwargs):
    """
    Name: tensor_tensor_mul()
    Purpose: Test function for multiplication between Tensor and Tensor.
    Parameters: **kwargs(dim1, dim2, np1, np2)
    Return: Bool

    Uses: \n
    Intended to be used inside of Testbench(func)
    """
    tensor_result = np.around((Tensor(kwargs['np1']) * Tensor(kwargs['np2'])).data, decimals=4)
    np_result = np.around(kwargs['np1'] * kwargs['np2'], decimals=4)
    return (tensor_result == np_result).all()

def tensor_scalar_mul(**kwargs):
    """
    Name: tensor_scalar_mul()
    Purpose: Test function for multiplication between Tensor and Scalar (integer/float).
    Parameters: **kwargs(dim1, dim2, np1, np2)
    Return: Bool

    Uses: \n
    Intended to be used inside of Testbench(func)
    """
    tensor_result = np.around((Tensor(kwargs['np1']) * kwargs['dim1']).data, decimals=4)
    np_result = np.around(kwargs['np1'] * kwargs['dim1'], decimals=4)
    return (tensor_result == np_result).all()

# import tensorflow as tf
# fashion_mnist = tf.keras.datasets.fashion_mnist

# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# train_images = train_images / 255.0

# test_images = test_images / 255.0

# model = tf.keras.Sequential([
#     tf.keras.layers.Flatten(input_shape=(28, 28)),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(10)
# ])

# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
# model.build()
# model.summary()

# model.fit(train_images, train_labels, epochs=10)

model = Model([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation=Activation.ReLU),
    Dense(10)
])

model.compile(loss="Loss", optimizer="Optimizer")
model.view()