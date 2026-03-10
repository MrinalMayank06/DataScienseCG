# Scenario Question

# A biomedical researcher is designing a convolutional neural network (CNN) to analyze microscopic grayscale images of blood cells (64×64 pixels).
# She begins with a single convolutional layer defined as follows:

# Input: 64×64 grayscale images (1 channel)
# Convolutional Layer: 16 filters, each of size 5×5
# Stride: 1
# Padding: Same
# Activation Function: ReLU
# The goal of this layer is to extract local structural features such as cell boundaries and texture variations, which will later help in
#  classifying whether the blood cells are healthy or show signs of abnormality.

import tensorflow as tf
from tensorflow import keras

conv = keras.layers.Conv2D(
    filters=16,
    kernel_size=(5,5),
    strides=(1,1),
    padding='same',
    activation='relu',
    input_shape=(64,64,1)
)

model = keras.Sequential([conv])
model.build(input_shape=(None,64,64,1))
model.summary()

print("Output Shape:", (None,64,64,16))
print("Weights:", (5,5,1,16))
print("Bias   :", (16,))
print("Total  :", 5*5*1*16 + 16)