
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

"""
Smart Waste Classification System
3-class image classification: Recyclable, Organic, Non-Recyclable
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import shutil
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.metrics import confusion_matrix, accuracy_score
import itertools

# Basic settings  
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_CNN = 1
EPOCHS_TL = 1

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print("\n" + "="*60)
print("SMART WASTE CLASSIFICATION SYSTEM")
print("="*60)
print(f"Working directory: {BASE_DIR}")

# Define all possible data sources
POSSIBLE_DATA_SOURCES = {
    # TrashNet dataset (most likely with actual images)
    "trashnet_resized": os.path.join(BASE_DIR, "trashnet-master", "data", "dataset-resized"),
    "trashnet_data": os.path.join(BASE_DIR, "trashnet-master", "data"),
    
    # Archive folders
    "archive_garbage": os.path.join(BASE_DIR, "archive (2)", "Garbage classification"),
    "archive1_dataset": os.path.join(BASE_DIR, "archive (1)", "DATASET"),
    "archive_original": os.path.join(BASE_DIR, "archive"),
    
    # One-indexed files folders (might contain actual images)
    "one_indexed_test": os.path.join(BASE_DIR, "archive (2)", "one-indexed-files-notrash_test.txt"),
}

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "validation")

# Target class names
CLASS_NAMES = ["recyclable", "organic", "non_recyclable"]

def find_image_folders():
    """Find all folders that actually contain images"""
    image_folders = []
    
    print("\n" + "-"*60)
    print("SCANNING FOR IMAGE FOLDERS")
    print("-"*60)
    
    # Common image extensions
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.JPG', '.JPEG', '.PNG']
    
    # Scan all possible locations
    scan_locations = [
        os.path.join(BASE_DIR, "trashnet-master", "data", "dataset-resized"),
        os.path.join(BASE_DIR, "trashnet-master", "data"),
        os.path.join(BASE_DIR, "archive (2)", "Garbage classification"),
        os.path.join(BASE_DIR, "archive (1)", "DATASET", "TRAIN"),
        os.path.join(BASE_DIR, "archive (1)", "DATASET", "TEST"),
        os.path.join(BASE_DIR, "archive", "Hazardous"),
        os.path.join(BASE_DIR, "archive", "Non-Recyclable"),
        os.path.join(BASE_DIR, "archive", "Organic"),
        os.path.join(BASE_DIR, "archive", "Recyclable"),
    ]
    
    for location in scan_locations:
        if os.path.exists(location):
            print(f"\nChecking: {location}")
            
            # If it's a directory, check its contents
            if os.path.isdir(location):
                # Check if this directory directly contains images
                dir_images = []
                for f in os.listdir(location):
                    f_path = os.path.join(location, f)
                    if os.path.isfile(f_path) and any(f.endswith(ext) for ext in image_exts):
                        dir_images.append(f)
                
                if dir_images:
                    print(f"  ✅ Found {len(dir_images)} images directly in this folder")
                    image_folders.append((location, "root", len(dir_images)))
                
                # Check subdirectories
                for subdir in os.listdir(location):
                    subdir_path = os.path.join(location, subdir)
                    if os.path.isdir(subdir_path):
                        image_count = 0
                        for f in os.listdir(subdir_path):
                            if any(f.endswith(ext) for ext in image_exts):
                                image_count += 1
                        
                        if image_count > 0:
                            print(f"  📁 {subdir}: {image_count} images")
                            image_folders.append((subdir_path, subdir.lower(), image_count))
    
    return image_folders

def organize_dataset_from_sources(image_folders):
    """Organize images from found sources into train/validation splits"""
    
    print("\n" + "-"*60)
    print("ORGANIZING DATASET")
    print("-"*60)
    
    # Create target directories
    for split in ['train', 'validation']:
        for cls in CLASS_NAMES:
            os.makedirs(os.path.join(DATASET_DIR, split, cls), exist_ok=True)
    
    # Map found folders to our target classes
    class_mapping = {
        'recyclable': ['recyclable', 'cardboard', 'glass', 'metal', 'paper', 'plastic', 'recyclable'],
        'organic': ['organic', 'organics', 'food'],
        'non_recyclable': ['non-recyclable', 'hazardous', 'trash', 'non recyclable', 'nonrecyclable', 'hazardous', 'non_rec', 'non-recyclable']
    }
    
    total_copied = 0
    
    for folder_path, folder_name, image_count in image_folders:
        # Determine target class
        target_class = None
        folder_lower = folder_name.lower()
        
        for cls, keywords in class_mapping.items():
            if any(keyword in folder_lower for keyword in keywords):
                target_class = cls
                break
        
        # If no match, check parent folder name
        if not target_class:
            parent_name = os.path.basename(os.path.dirname(folder_path)).lower()
            for cls, keywords in class_mapping.items():
                if any(keyword in parent_name for keyword in keywords):
                    target_class = cls
                    break
        
        # Default to recyclable if still no match
        if not target_class:
            target_class = 'recyclable'
        
        print(f"\nProcessing: {folder_path}")
        print(f"  → Mapping to: {target_class}")
        
        # Get all image files
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
            image_files.extend(list(Path(folder_path).glob(f'*{ext}')))
            image_files.extend(list(Path(folder_path).glob(f'*{ext.upper()}')))
        
        if not image_files:
            continue
        
        # Shuffle and split (80-20)
        random.shuffle(image_files)
        split_idx = int(0.8 * len(image_files))
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]
        
        # Copy files
        for img_file in train_files:
            dest = os.path.join(TRAIN_DIR, target_class, img_file.name)
            shutil.copy2(img_file, dest)
            total_copied += 1
        
        for img_file in val_files:
            dest = os.path.join(VAL_DIR, target_class, img_file.name)
            shutil.copy2(img_file, dest)
            total_copied += 1
        
        print(f"  ✅ Copied {len(train_files)} train, {len(val_files)} val")
    
    return total_copied

def count_images(folder_path):
    """Count images in a folder"""
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".JPG", ".JPEG", ".PNG")
    if not os.path.exists(folder_path):
        return 0
    try:
        return sum(1 for f in os.listdir(folder_path) if f.endswith(exts))
    except:
        return 0

def show_dataset_info():
    """Display dataset statistics"""
    print("\n" + "-"*60)
    print("DATASET STATISTICS")
    print("-"*60)
    
    total_train = 0
    total_val = 0
    
    print("\nTRAIN SET:")
    for cls in CLASS_NAMES:
        cls_path = os.path.join(TRAIN_DIR, cls)
        count = count_images(cls_path)
        total_train += count
        print(f"  📁 {cls}: {count} images")
    print(f"  Total: {total_train} images")
    
    print("\nVALIDATION SET:")
    for cls in CLASS_NAMES:
        cls_path = os.path.join(VAL_DIR, cls)
        count = count_images(cls_path)
        total_val += count
        print(f"  📁 {cls}: {count} images")
    print(f"  Total: {total_val} images")
    
    return total_train, total_val

# Main execution
print("\n" + "="*60)
print("STARTING WASTE CLASSIFICATION SYSTEM")
print("="*60)

# Step 1: Find all image folders
image_folders = find_image_folders()

if not image_folders:
    print("\n❌ No image folders found!")
    print("\nCreating dummy data for testing...")
    
    # Create dummy data
    for cls in CLASS_NAMES:
        os.makedirs(os.path.join(TRAIN_DIR, cls), exist_ok=True)
        os.makedirs(os.path.join(VAL_DIR, cls), exist_ok=True)
        
        # Create dummy images
        for i in range(5):
            dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            import cv2
            cv2.imwrite(os.path.join(TRAIN_DIR, cls, f"dummy_{i}.jpg"), dummy_img)
            cv2.imwrite(os.path.join(VAL_DIR, cls, f"dummy_{i}.jpg"), dummy_img)
    
    print("✅ Created dummy images for testing")
else:
    print(f"\n✅ Found {len(image_folders)} folders with images")
    
    # Step 2: Organize dataset
    total_copied = organize_dataset_from_sources(image_folders)
    print(f"\n✅ Successfully organized {total_copied} images")

# Step 3: Show dataset statistics
train_count, val_count = show_dataset_info()

if train_count == 0:
    print("\n❌ No training images available. Exiting...")
    exit()

# Step 4: Data Augmentation
print("\n" + "-"*60)
print("DATA PREPROCESSING")
print("-"*60)

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
], name="augmentation")

normalizer = layers.Rescaling(1./255)

# Load datasets
print("\nLoading datasets...")

try:
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    # Performance optimization
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    print("\n✅ Data preprocessing pipeline ready!")
    
except Exception as e:
    print(f"\n❌ Error loading datasets: {e}")
    exit()

# Step 5: Build and train Custom CNN
print("\n" + "-"*60)
print("TRAINING CUSTOM CNN")
print("-"*60)

def build_custom_cnn(num_classes=3):
    model = keras.Sequential([
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        data_augmentation,
        normalizer,
        
        layers.Conv2D(32, (3,3), activation="relu", padding='same'),
        layers.MaxPooling2D(),
        
        layers.Conv2D(64, (3,3), activation="relu", padding='same'),
        layers.MaxPooling2D(),
        
        layers.Conv2D(128, (3,3), activation="relu", padding='same'),
        layers.MaxPooling2D(),
        
        layers.Conv2D(256, (3,3), activation="relu", padding='same'),
        layers.MaxPooling2D(),
        
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])
    return model

cnn_model = build_custom_cnn(len(CLASS_NAMES))

cnn_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nTraining Custom CNN...")
history_cnn = cnn_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_CNN,
    verbose=1
)

# Step 6: Build and train Transfer Learning model
print("\n" + "-"*60)
print("TRAINING TRANSFER LEARNING MODEL")
print("-"*60)

def build_transfer_model(num_classes=3):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights="imagenet"
    )
    
    base_model.trainable = False
    
    inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = keras.Model(inputs, outputs)
    return model

tl_model = build_transfer_model(len(CLASS_NAMES))

tl_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nTraining Transfer Learning Model...")
history_tl = tl_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_TL,
    verbose=1
)

# Step 7: Compare results
print("\n" + "="*60)
print("RESULTS COMPARISON")
print("="*60)

cnn_final_val = history_cnn.history["val_accuracy"][-1]
tl_final_val = history_tl.history["val_accuracy"][-1]

print(f"\n Custom CNN Validation Accuracy: {cnn_final_val:.4f} ({cnn_final_val*100:.2f}%)")
print(f" Transfer Learning Validation Accuracy: {tl_final_val:.4f} ({tl_final_val*100:.2f}%)")

print("\n" + "-"*60)
if cnn_final_val > tl_final_val:
    print(" Custom CNN performed better!")
elif tl_final_val > cnn_final_val:
    print(" Transfer Learning performed better!")
else:
    print(" Both models performed equally!")
print("-"*60)

print("\n" + "="*60)
print(" WASTE CLASSIFICATION SYSTEM COMPLETED!")
print("="*60)