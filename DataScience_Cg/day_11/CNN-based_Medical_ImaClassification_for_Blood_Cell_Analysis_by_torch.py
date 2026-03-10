# This scenario represents a CNN-based medical image classification system used for detecting abnormal blood cells from microscopic images.
# Scenario Question

# A biomedical researcher is designing a convolutional neural network (CNN) to analyze microscopic grayscale images of blood cells (64×64 pixels). She begins with a single convolutional layer defined as follows:

# Input: 64×64 grayscale images (1 channel)
# Convolutional Layer: 16 filters, each of size 5×5
# Stride: 1
# Padding: Same
# Activation Function: ReLU
# The goal of this layer is to extract local structural features such as cell boundaries and texture variations, which will later help in classifying whether the blood cells are healthy or show signs of abnormality.


# Domain
# ➡ Biomedical Image Analysis

# Task Type
# ➡ Image Classification

# Model Type
# ➡ Convolutional Neural Network (CNN)

# Input Data
# ➡ Microscopic grayscale blood cell images

# Goal
# # ➡ Healthy vs Abnormal Cell Detection


import torch
import torch.nn as nn

conv = nn.Conv2d(
    in_channels=1,
    out_channels=16,
    kernel_size=5,
    stride=1,
    padding=2
)

relu = nn.ReLU()

x = torch.randn(1, 1, 64, 64)

out = conv(x)
out = relu(out)

print(out.shape)

print("Weights:", conv.weight.shape)
print("Bias:", conv.bias.shape)

total_params = 16*1*5*5 + 16
print("Total Parameters:", total_params)