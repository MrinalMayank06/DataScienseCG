# Scenario: AI System for Detecting Lung Diseases from Chest X-rays
# 🚨 The Problem

# A large hospital receives thousands of chest X-rays daily.

# Radiologists are:

# Overworked

# Limited in number

# Required to make fast decisions

# Sometimes critical conditions like:

# 👉 Pneumonia
# 👉 COVID-19
# 👉 Normal lungs

# must be identified within minutes.

# Delays can cost lives.

# 💡 The Solution: Build an AI Assistant

# The hospital decides to deploy an AI model that can pre-screen X-rays and
# alert doctors.

# But there is a challenge…

# ❗ Medical datasets are usually SMALL.

# Training a deep neural network from scratch would require:

# Millions of labeled X-rays

# Massive GPU clusters

# Months of training

# Not practical.

# ⭐ Enter Transfer Learning (Your Code)

# Instead of starting from zero, engineers use:

# 👉 ResNet50 trained on ImageNet

# Although ImageNet contains everyday objects (dogs, cars, etc.),
# the early CNN layers learn universal visual patterns, like:

# ✅ Edges
# ✅ Gradients
# ✅ Textures
# ✅ Shapes

# These features are also present in medical scans.

# Import PyTorch library (main deep learning framework)
import torch

# Import torchvision models (contains pre-trained CNN architectures)
import torchvision.models as models

# Import neural network module from PyTorch
from torch import nn


# -------------------------------------------------------
# Step 1: Load a Pre-trained ResNet50 Model
# -------------------------------------------------------

# ResNet50 is a deep convolutional neural network with 50 layers.
# It is already trained on the ImageNet dataset (1.2 million images, 1000 classes).
# Using a pre-trained model allows us to reuse learned visual features
# such as edges, textures, shapes, and patterns.

model = models.resnet50(pretrained=True)


# -------------------------------------------------------
# Step 2: Freeze All Pre-trained Layers
# -------------------------------------------------------

# Transfer learning strategy:
# We freeze earlier layers so their weights do not change during training.
# These layers already learned general image features.

for param in model.parameters():
    param.requires_grad = False

# requires_grad = False means:
# - Gradients will NOT be computed
# - Weights will NOT be updated during backpropagation
# - Training becomes faster and needs less data


# -------------------------------------------------------
# Step 3: Modify the Final Classification Layer
# -------------------------------------------------------

# The original ResNet50 final layer predicts 1000 classes (ImageNet).
# Our problem only has 3 classes:
# 1. Normal
# 2. Pneumonia
# 3. COVID-19

num_classes = 3

# Replace the last fully connected layer (fc)
# model.fc.in_features gives the number of input features to the layer
# (2048 for ResNet50)

model.fc = nn.Linear(model.fc.in_features, num_classes)

# Now the model output will be 3 neurons instead of 1000.


# -------------------------------------------------------
# Step 4: Train Only the New Layer
# -------------------------------------------------------

# Since earlier layers are frozen,
# only the new fully connected layer will be trained.

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Trainable params: {trainable_params}")


# -------------------------------------------------------
# Expected Output Explanation
# -------------------------------------------------------

# The output will show the number of parameters that are trainable.
# Because we froze the backbone network, only the new classification
# layer parameters will be updated during training.

# This is a typical Transfer Learning workflow used in:
# - Medical Image Analysis
# - Face Recognition
# - Object Detection
# - Small dataset problems