import numpy as np
import matplotlib.pyplot as plt

# Define activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def tanh(z):
    return np.tanh(z)

def relu(z):
    return np.maximum(0, z)

def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

# Forward pass functions
def forward_pass(X, weights, biases, activation_function):
    z = np.dot(weights, X) + biases
    if activation_function == 'sigmoid':
        return sigmoid(z)
    elif activation_function == 'tanh':
        return tanh(z)
    elif activation_function == 'relu':
        return relu(z)
    elif activation_function == 'softmax':
        return softmax(z)
    else:
        raise ValueError("Unknown activation function")
    
# Example inputs
X = np.array([[0.5], [0.8]])
weights = np.array([[0.2, 0.4], [0.6, 0.1]])
biases = np.array([[0.1], [0.2]])

# Perform forward pass with different activation functions
output_sigmoid = forward_pass(X, weights, biases, 'sigmoid')
output_tanh = forward_pass(X, weights, biases, 'tanh')
output_relu = forward_pass(X, weights, biases, 'relu')
output_softmax = forward_pass(X, weights, biases, 'softmax')

print("Output with Sigmoid Activation:\n", output_sigmoid)
print("Output with Tanh Activation:\n", output_tanh)
print("Output with ReLU Activation:\n", output_relu)
print("Output with Softmax Activation:\n", output_softmax)

# Define range of inputs
z = np.linspace(-10, 10, 400)

# Compute outputs for each activation function
output_sigmoid = sigmoid(z)
output_tanh = tanh(z)
output_relu = relu(z)
output_softmax = softmax(z)

# Plot the activation functions
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(z, output_sigmoid, label='Sigmoid')
plt.title('Sigmoid Activation Function')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(z, output_tanh, label='Tanh', color='orange')
plt.title('Tanh Activation Function')
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(z, output_relu, label='ReLU', color='green')
plt.title('ReLU Activation Function')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(z, output_softmax, label='Softmax', color='red')
plt.title('Softmax Activation Function')
plt.grid(True)

plt.tight_layout()
plt.show()