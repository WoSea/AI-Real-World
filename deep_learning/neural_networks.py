# pip install tensorflow
# pip install torch torchvision
from tensorflow.keras.datasets import mnist, cifar10
import tensorflow as tf
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Load MNIST
mnist_data = mnist.load_data()
X_train_mnist, y_train_mnist = mnist_data[0]
X_test_mnist, y_test_mnist = mnist_data[1]
print("MNIST data shapes:")
print("X_train:", X_train_mnist.shape)
print("X_test:", X_test_mnist.shape)

# Load CIFAR-10
cifar10_data = cifar10.load_data()
X_train_cifar10, y_train_cifar10 = cifar10_data[0]
X_test_cifar10, y_test_cifar10 = cifar10_data[1]
print("CIFAR-10 data shapes:")
print("X_train:", X_train_cifar10.shape)
print("X_test:", X_test_cifar10.shape)

# Define a basic dense layer
layer = tf.keras.layers.Dense(units=10, activation='relu')
print("TensorFlow Layer:", layer)

layer = nn.Linear(in_features=10, out_features=5)
print("Pytorch layer:", layer)

# Define a basic dense layer
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.summary()

# Visualize MNIST sample
plt.imshow(X_train_mnist[0], cmap='gray')
plt.title("Sample MNIST Image")
plt.axis('off')
plt.show()

# Visualize CIFAR-10 sample
plt.imshow(X_train_cifar10[0])
plt.title("Sample CIFAR-10 Image")
plt.axis('off')
plt.show()