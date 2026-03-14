# Scenario 1: Smart Waste Classification System (Waste Image Classifier)

# What we are building:
# City wants an AI system that looks at a waste image and outputs exactly 1 label:
# 1) Recyclable
# 2) Organic
# 3) Non-Recyclable
# This is a 3-class image classification problem.

# How this works (simple flow):
# Step 1: Collect images and store them in folders class-wise (train/validation split).
# Step 2: Convert images into model-friendly tensors:
#         - Resize images (example: 224x224)
#         - Normalize pixel values (0-255 -> 0-1)
# Step 3: Use data augmentation so the model doesn’t overfit:
#         - Rotation, Flip, Zoom, Brightness/Contrast etc.
# Step 4: Train a Custom CNN (Conv + Pooling + Dense + Softmax).
# Step 5: Evaluate model performance:
#         - Accuracy score
#         - Confusion Matrix
#         - Sample predictions on test images
# Step 6: Train a Transfer Learning model (MobileNetV2 / ResNet50 / VGG16):
#         - Load pretrained base model
#         - Freeze base layers
#         - Add custom classification layers
#         - Train on our dataset
# Step 7: Compare results:
#         - Custom CNN accuracy vs Transfer Learning accuracy

# Tech used:
# - Python (main programming language)
# - TensorFlow / Keras (building + training deep learning models)
# - Matplotlib (graphs like accuracy/loss + visual results)
# - NumPy (arrays, preprocessing support)
# - Scikit-learn (confusion matrix + accuracy calculation)


# TASK 1 – Dataset Collection

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset_path = r"C:\Users\krish\Downloads\Garbage classification"

print("Dataset Path:", dataset_path)

classes = os.listdir(dataset_path)

print("\nDataset Classes:")
print(classes)

print("\nNumber of Images in Each Class:")

for c in classes:
    path = os.path.join(dataset_path, c)
    print(c, ":", len(os.listdir(path)))


# TASK 2 – Data Preprocessing

from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    
    rescale=1./255,
    validation_split=0.2,
    
    rotation_range=25,
    horizontal_flip=True,
    zoom_range=0.2,
    brightness_range=[0.8,1.2]
)

train_generator = datagen.flow_from_directory(
    
    dataset_path,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    
    dataset_path,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

print("\nClass Mapping:", train_generator.class_indices)


# TASK 3 – CNN Model Development

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential()

model.add(Conv2D(32,(3,3),activation='relu',input_shape=(224,224,3)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(train_generator.num_classes,activation='softmax'))

model.compile(
    
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    
    train_generator,
    validation_data=validation_generator,
    epochs=3
)


# TASK 3 – Accuracy Graph

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend(["Train","Validation"])

plt.show()


# TASK 3 – Loss Graph

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend(["Train","Validation"])

plt.show()


# TASK 4 – Model Evaluation

from sklearn.metrics import confusion_matrix, accuracy_score

pred = model.predict(validation_generator)

y_pred = np.argmax(pred,axis=1)

cm = confusion_matrix(validation_generator.classes,y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(cm,annot=True,cmap="Blues")

plt.title("Confusion Matrix")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

acc = accuracy_score(validation_generator.classes,y_pred)

print("\nModel Accuracy:",acc)


# TASK 4 – Sample Prediction 

from tensorflow.keras.preprocessing import image
import random

random_class = random.choice(classes)

class_folder = os.path.join(dataset_path, random_class)

images = [f for f in os.listdir(class_folder) if f.endswith(('.jpg','.jpeg','.png'))]

img_name = random.choice(images)

img_path = os.path.join(class_folder, img_name)

print("\nTesting Image:", img_path)

img = image.load_img(img_path,target_size=(224,224))

plt.imshow(img)
plt.axis("off")
plt.show()

img_array = image.img_to_array(img)/255.0
img_array = np.expand_dims(img_array,axis=0)

prediction = model.predict(img_array)

class_names = list(train_generator.class_indices.keys())

print("Predicted Class:",class_names[np.argmax(prediction)])


# TASK 5 – Transfer Learning

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model

base_model = MobileNetV2(
    
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(128,activation='relu')(x)

x = Dropout(0.5)(x)

predictions = Dense(train_generator.num_classes,activation='softmax')(x)

transfer_model = Model(inputs=base_model.input,outputs=predictions)

transfer_model.compile(

    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_transfer = transfer_model.fit(

    train_generator,
    validation_data=validation_generator,
    epochs=3
)


# TASK 6 – Accuracy Comparison

cnn_acc = max(history.history['val_accuracy'])

transfer_acc = max(history_transfer.history['val_accuracy'])

print("\nCustom CNN Accuracy:",cnn_acc)

print("Transfer Learning Accuracy:",transfer_acc)