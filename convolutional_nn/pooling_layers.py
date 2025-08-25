import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Create simple 6x6 image
image = np.array([
    [1, 2, 3, 0, 1, 2],
    [4, 5, 6, 1, 2, 3],
    [7, 8, 9, 0, 1, 2],
    [1, 2, 3, 4, 5, 6],
    [7, 8, 9, 1, 2, 3],
    [4, 5, 6, 7, 8, 9]
], dtype=np.float32)

# Reshape to tensor (batch, h, w, c)
image_tensor = image.reshape(1, 6, 6, 1)

# Max Pooling 2x2
max_pool = tf.nn.max_pool2d(image_tensor, ksize=2, strides=2, padding="VALID")

# Average Pooling 2x2
avg_pool = tf.nn.avg_pool2d(image_tensor, ksize=2, strides=2, padding="VALID")

# Convert to numpy
max_pool_result = max_pool.numpy().squeeze()
avg_pool_result = avg_pool.numpy().squeeze()

# Plot results
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("Original 6x6")
plt.imshow(image, cmap="Blues")
plt.colorbar()

plt.subplot(1,3,2)
plt.title("Max Pooling (2x2)")
plt.imshow(max_pool_result, cmap="Blues")
plt.colorbar()

plt.subplot(1,3,3)
plt.title("Average Pooling (2x2)")
plt.imshow(avg_pool_result, cmap="Blues")
plt.colorbar()

plt.show()
