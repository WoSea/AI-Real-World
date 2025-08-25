import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# Create simple grayscale image (gradient + edge)
image = np.array([
    [3, 3, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [3, 3, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3]
], dtype=np.float32)

# Horizontal edge detection filter (Sobel-like)
filter_horizontal = np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]
], dtype=np.float32)

# Vertical edge detection filter
filter_vertical = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
], dtype=np.float32)

# Convolution
out_horizontal = convolve2d(image, filter_horizontal, mode="valid")
out_vertical = convolve2d(image, filter_vertical, mode="valid")

# Plot results
fig, axs = plt.subplots(1, 4, figsize=(14,4))
axs[0].imshow(image, cmap="gray")
axs[0].set_title("Original Image")
axs[0].axis("off")

axs[1].imshow(filter_horizontal, cmap="gray")
axs[1].set_title("Horizontal Filter")
axs[1].axis("off")

axs[2].imshow(out_horizontal, cmap="gray")
axs[2].set_title("Detected Horizontal Edges")
axs[2].axis("off")

axs[3].imshow(out_vertical, cmap="gray")
axs[3].set_title("Detected Vertical Edges")
axs[3].axis("off")

plt.show()

# TensorFlow version
import tensorflow as tf
import numpy as np

# Input: one channel (grayscale) 5x5 image
X = np.array([[[[1], [2], [3], [0], [1]],
               [[0], [1], [2], [3], [1]],
               [[1], [0], [1], [2], [2]],
               [[2], [1], [0], [1], [3]],
               [[3], [2], [1], [0], [0]]]], dtype=np.float32)  # shape (1, 5, 5, 1)

# Create layer Conv2D (1 filter 3x3)
conv = tf.keras.layers.Conv2D(filters=1, kernel_size=(3,3), strides=1, padding="valid")

# Apply Conv layer
y = conv(X)

print("Output feature map shape:", y.shape)
print("Learned filter (weights):")
print(conv.weights[0].numpy().reshape(3,3))
print("Bias:", conv.weights[1].numpy())
print("Output feature map:\n", y.numpy()[0, :, :, 0])

# Pytorch version
import torch
import torch.nn as nn

# Input: batch=1, channel=1, H=5, W=5
X = torch.tensor([[[[1,2,3,0,1],
                    [0,1,2,3,1],
                    [1,0,1,2,2],
                    [2,1,0,1,3],
                    [3,2,1,0,0]]]], dtype=torch.float32)

# Tạo Conv2d (1 input channel, 1 output channel, kernel 3x3)
conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0)

# Apply Conv
y = conv(X)

print("Output feature map shape:", y.shape)
print("Learned filter (weights):")
print(conv.weight.data[0,0])
print("Bias:", conv.bias.data)
print("Output feature map:\n", y[0,0].detach().numpy())
