# Music Genre Classification using Transfer Learning (ResNet50)
# Spectrogram images already available (no librosa used)

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

# --------------------------------
# 1. Dataset Path
# --------------------------------

spectrogram_dataset = "spectrogram_dataset"

# Folder structure assumed
# spectrogram_dataset/
#     rock/
#     jazz/
#     classical/
#     hiphop/
#     electronic/

# --------------------------------
# 2. Image Data Generator
# --------------------------------

img_size = (224,224)
batch_size = 32

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    spectrogram_dataset,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    spectrogram_dataset,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# --------------------------------
# 3. Load Pretrained ResNet50
# --------------------------------

base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False


# --------------------------------
# 4. Custom Classification Layers
# --------------------------------

x = base_model.output
x = GlobalAveragePooling2D()(x)

x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)

output = Dense(5, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# --------------------------------
# 5. Compile Model
# --------------------------------

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --------------------------------
# 6. Train Model
# --------------------------------

model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

# --------------------------------
# 7. Fine Tuning (Unfreeze Top Layers)
# --------------------------------

for layer in base_model.layers[-20:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

# --------------------------------
# 8. Evaluate Model
# --------------------------------

loss, accuracy = model.evaluate(val_generator)

print("Validation Accuracy:", accuracy)