import numpy as np
import random
import time

from grand.tensor import Tensor
from grand.layers import Layer, Dense, Flatten
from grand.nn import Model
from grand.activations import Activation

from grand.utils import Colors

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

# model = Model([
#     Flatten(input_shape=(28, 28)),
#     Dense(128, activation=Activation.ReLU),
#     Dense(10)
# ])

# model.compile(loss="Loss", optimizer="Optimizer")
# model.view()

from grand import gcuda

print(gcuda.add(1, 2))