from grand.tensor import Tensor
from grand.layers import Layer, Input, Dense
from grand.nn import Model

import tensorflow as tf
from tensorflow import keras
import numpy as np

t1 = Tensor([1, 2, 3])
print(t1.dtype)
# t2 = Tensor([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])


# model = Model([
#     Input(4, 3),
#     Dense(4),
#     Dense(1)
#     ])

# model.forward()

# Input(batch_size (number of data points), data size)
# Dense(data size, output size)
# etc...

# print(tf.config.list_physical_devices())

# mnist = keras.datasets.mnist
# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# print("Training initial shape: ", train_images.shape)

# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(5, activation='relu', input_shape=(100, 5)),
#     tf.keras.layers.Dense(10)
# ])

# model.compile(optimizer='adam', 
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

# model.build()
# model.summary()

# history = model.fit(train_images, train_labels, epochs=5)