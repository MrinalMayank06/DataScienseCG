#  (Simple feedforward LM with Keras)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Flatten

# Vocabulary size and embedding dimension
vocab_size = 50
embed_dim = 8

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=2), # context length=2
    Flatten(),
    Dense(32, activation='relu'),
    Dense(vocab_size, activation='softmax')  # predict next word
])

model.compile(optimizer='adam', loss='categorical_crossentropy')
model.build(input_shape=(None, 2))  # batch size flexible, sequence length = 2

model.summary()