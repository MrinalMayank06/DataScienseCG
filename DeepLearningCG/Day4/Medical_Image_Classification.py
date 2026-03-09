#  Scenario: Medical Image Classification
# You’re training a convolutional neural network (CNN) to detect pneumonia from chest X-rays.
# - Training accuracy: 95%
# - Validation accuracy: 74%
# At first glance, the model seems powerful — it almost perfectly classifies the training set. But the sharp drop in validation accuracy signals overfitting: the network has memorized the training images (specific pixel patterns, noise, or even hospital-specific artifacts) instead of learning generalizable features of pneumonia.

# ⚙️ Levers to Address Overfitting
# - Data Augmentation: Rotate, flip, and adjust brightness of X-rays to simulate variability.
# - Regularization: Apply dropout in dense layers or L2 weight decay.
# - Transfer Learning: Use a pretrained backbone (e.g., ResNet) to leverage generalized features.
# - Cross-validation: Ensure robustness across different patient subsets.
# - Early Stopping: Halt training when validation loss stops improving.


# Pneumonia Detection using CNN + Transfer Learning
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import os

# -------------------------------
# Configuration
# -------------------------------
DATASET_PATH = r"C:\Users\krish\Downloads\chest_xray"   # Should contain train/ and val/
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_PHASE1 = 20
EPOCHS_PHASE2 = 10
PATIENCE = 5

# -------------------------------
# Data Generators with Augmentation
# -------------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "val"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

print(f"Found {train_generator.samples} training images, {val_generator.samples} validation images.")

# -------------------------------
# Transfer Learning Base Model
# -------------------------------
base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze all base layers initially
base_model.trainable = False

# -------------------------------
# Custom Classifier Head
# -------------------------------
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu", kernel_regularizer=l2(0.001))(x)
x = Dropout(0.5)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------------
# Callbacks
# -------------------------------
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=PATIENCE,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "best_model_phase1.h5",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max"
)

# -------------------------------
# Phase 1: Train the top layers
# -------------------------------
print("\n--- Phase 1: Training top layers ---")
history1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_PHASE1,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# -------------------------------
# Phase 2: Fine-tune last 20 layers of ResNet50
# -------------------------------
# Unfreeze the last 20 layers
for layer in base_model.layers[-20:]:
    layer.trainable = True

# Recompile with very low learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Save best model during fine-tuning
checkpoint_ft = ModelCheckpoint(
    "best_model_phase2.h5",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max"
)

print("\n--- Phase 2: Fine-tuning ---")
history2 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_PHASE2,
    callbacks=[early_stop, checkpoint_ft],
    verbose=1
)

# -------------------------------
# Final Evaluation
# -------------------------------
loss, acc = model.evaluate(val_generator, verbose=0)
print(f"\n✅ Final Validation Accuracy: {acc:.4f}")
print(f"   Final Validation Loss: {loss:.4f}")

# -------------------------------
# Plot training curves
# -------------------------------
def plot_history(history, title):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title(title + ' - Accuracy')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title + ' - Loss')
    plt.legend()
    plt.show()

plot_history(history1, "Phase 1")
plot_history(history2, "Phase 2 (Fine-tuning)")

# Save final model
model.save("pneumonia_detector_final.h5")
print("Model saved as pneumonia_detector_final.h5")