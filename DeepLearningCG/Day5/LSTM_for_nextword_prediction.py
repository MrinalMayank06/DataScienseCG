#Code Demo (LSTM for next‑word prediction)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import numpy as np

# Vocabulary size and embedding dimension
vocab_size = 100
embed_dim = 16
seq_length = 5  # input sequence length

# Define the model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=seq_length),
    LSTM(64),  # LSTM layer with 64 units
    Dense(vocab_size, activation='softmax')  # predict next word
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Force the model to build by providing input shape
model.build(input_shape=(None, seq_length))

# Show the summary
model.summary()

# Optional: run a dummy prediction to confirm
dummy_input = np.random.randint(0, vocab_size, (1, seq_length))
print("Dummy prediction shape:", model.predict(dummy_input).shape)