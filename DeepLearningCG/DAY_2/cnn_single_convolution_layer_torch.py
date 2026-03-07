 # Scenario Question
# A machine learning engineer is designing a convolutional neural network (CNN) to process grayscale images of handwritten digits
#  (28×28 pixels). She starts with a single convolutional layer defined as follows

import torch
import torch.nn as nn

# Single Conv2D layer
conv = nn.Conv2d(
    in_channels  = 1,    # grayscale input
    out_channels = 32,   # 32 filters -> 32 feature maps
    kernel_size  = 3,    # 3x3 filter
    stride       = 1,
    padding      = 1     # SAME padding: output = same size
)

# Forward pass
x   = torch.randn(1, 1, 28, 28) # (batch, C, H, W)
out = conv(x)
print(out.shape) # torch.Size([1, 32, 28, 28])

# Inspect learnable params
print('Weights:', conv.weight.shape) # (32, 1, 3, 3)
print('Bias   :', conv.bias.shape)   # (32,)
print('Total  :', 32*1*3*3 + 32)     # = 320  : bhai iska topic file name batao 