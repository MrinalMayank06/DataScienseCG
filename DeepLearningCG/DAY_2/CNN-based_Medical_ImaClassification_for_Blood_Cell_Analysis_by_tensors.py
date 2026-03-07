# Conv → ReLU → MaxPooling → Conv → ReLU → Flatten → Dense → Classification

import tensorflow as tf
from tensorflow import keras

# Single Conv2D layer
conv = keras.layers.Conv2D(
    filters     = 32,
    kernel_size = (3, 3),
    strides     = (1, 1),
    padding     = 'same',
    activation  = 'relu',
    input_shape = (28, 28, 1)
)

# Build and summarise
model = keras.Sequential([conv])
model.build(input_shape=(None, 28, 28, 1))
model.summary()
# Output shape: (None, 28, 28, 32)
# Trainable params: 320

