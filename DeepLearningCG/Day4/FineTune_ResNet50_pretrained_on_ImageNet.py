#Fine tuning a pretrained model
# Scenario: Fine-tune ResNet-50 pretrained on ImageNet
#  to classify 5 types of plant disease from 3,000 leaf images.

# Import PyTorch main library
# PyTorch is used for building and training deep learning models
import torch

# Import torchvision which contains datasets, pretrained models and image
#  utilities
import torchvision

# Import neural network module from PyTorch
# nn contains layers like Linear, Conv2D, ReLU etc.
import torch.nn as nn

# Import pretrained models and image transformations
from torchvision import models, transforms

# Import DataLoader to load dataset in batches
from torch.utils.data import DataLoader

# Import ImageFolder dataset loader
# This loads images arranged in folder structure
# Example:
# dataset/
#     disease1/
#     disease2/
#     disease3/
from torchvision.datasets import ImageFolder



# STEP 1 : Load a Pretrained ResNet-50 Model



# ResNet50 is a deep CNN with 50 layers designed by Microsoft
# It is trained on ImageNet dataset (~1.2 million images, 1000 classes)


# Using pretrained weights helps the model already know
# basic visual features like edges, shapes, textures etc.

model = models.resnet50(weights='IMAGENET1K_V2')

# weights='IMAGENET1K_V2'
# loads pretrained weights trained on ImageNet
# This technique is called TRANSFER LEARNING

# Instead of training from scratch,
# we reuse knowledge learned from a huge dataset.



# STEP 2 : Freeze ALL layers


# Freezing means we stop updating weights during training

# Why freeze?
# Because the early layers already learned useful
# general visual features like:
# edges
# corners
# textures
# patterns

# We only want to train the final layers for our dataset.

for param in model.parameters():
    param.requires_grad = False

# requires_grad = False means:
# PyTorch will NOT compute gradients for these parameters
# during backpropagation.

# Result:
# The pretrained weights stay unchanged.



# STEP 3 : Replace the classifier head (Fully Connected layer)


# ResNet originally outputs 1000 classes (ImageNet classes)
# But our plant disease dataset has only 5 classes.

num_classes = 5

# The final layer of ResNet is:
# model.fc

# We replace it with a custom classifier.

model.fc = nn.Sequential(

    # Dropout randomly disables neurons during training
    # Helps reduce overfitting
    nn.Dropout(0.4),

    # First fully connected layer
    # model.fc.in_features gives the number of input features
    # from the previous ResNet layer (usually 2048)

    nn.Linear(model.fc.in_features, 256),

    # ReLU activation introduces non-linearity
    # Helps model learn complex patterns
    nn.ReLU(),

    # Final layer outputs probabilities for each class
    nn.Linear(256, num_classes)
)

# Architecture of new classifier:
#
# Feature Vector (2048)
#        ↓
#     Dropout
#        ↓
#   Linear (2048 → 256)
#        ↓
#       ReLU
#        ↓
#   Linear (256 → 5 classes)



# STEP 4 : Unfreeze last residual block (Fine-Tuning)


# ResNet architecture is divided into blocks:
#
# conv1
# layer1
# layer2
# layer3
# layer4
# fc

# layer4 is the deepest feature extraction block.

# We unfreeze it so the network can slightly adjust
# high-level features for plant diseases.

for param in model.layer4.parameters():
    param.requires_grad = True

# This technique is called:
# FINE-TUNING

# Strategy used here:
#
# Early layers  → Frozen (generic features)
# Deep layers   → Trainable (task-specific features)
# Classifier    → Fully trainable

 
# STEP 5 : Differential Learning Rates
 

# Different parts of the network learn at different speeds.

# Pretrained layers need SMALL learning rate
# because we only want small adjustments.

# New classifier layers need HIGHER learning rate
# because they are randomly initialized.

optimizer = torch.optim.Adam([

    # Lower learning rate for pretrained layer4
    {'params': model.layer4.parameters(), 'lr': 1e-4},

    # Higher learning rate for new classifier head
    {'params': model.fc.parameters(), 'lr': 1e-3},
])

# Adam optimizer:
# Adaptive learning rate optimization algorithm
# Combines benefits of Momentum and RMSProp

# Learning rates:
#
# layer4 → 0.0001 (small updates)
# fc     → 0.001  (faster learning)


# ---------------------------------------------------------
# STEP 6 : Loss Function
# ---------------------------------------------------------

# CrossEntropyLoss is used for multi-class classification.

criterion = nn.CrossEntropyLoss()

# It combines:
#
# LogSoftmax + Negative Log Likelihood

# Expected input:
#
# Model output → raw logits
# Target → class index (0,1,2,3,4)

# Example:
#
# Output = [2.1, -1.2, 0.5, 3.4, -0.7]
# Target = 3



# FINAL TRAINING FLOW


# During training the process is:

#  Input image
#        ↓
#  ResNet pretrained layers (mostly frozen)
#        ↓
#  Layer4 (fine-tuned)
#        ↓
#  Custom classifier (fc)
#        ↓
#  Output logits for 5 plant diseases
#        ↓
#  CrossEntropyLoss calculates error
#        ↓
#  Backpropagation
#        ↓
#  Only layer4 + fc weights update

print("Model Loaded Successfully\n")

print(model)

print("\nTrainable Parameters:\n")

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)