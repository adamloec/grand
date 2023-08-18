from grand.tensor import Tensor
from grand.layers import Layer, Input, Dense
from grand.nn import Model

# t1 = Tensor([1, 2, 3])
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

import tensorflow as tf
from tensorflow import keras
import numpy as np

# print(tf.config.list_physical_devices())

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print("Training initial shape: ", train_images.shape)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam', 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

model.summary()

# history = model.fit(train_images, train_labels, epochs=5)