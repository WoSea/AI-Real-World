import numpy as np
import matplotlib.pyplot as plt

# Mean Squared Error Loss Function
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Binary Cross-Entropy
def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-12  # Small value to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Example data
y_true = np.array([1, 0, 1, 1])
y_pred = np.array([0.8, 0.2, 0.6, 0.9])

# Calculate loss
mse = mean_squared_error(y_true, y_pred)
bce = binary_cross_entropy(y_true, y_pred)

print(f"Mean Squared Error:{mse:.4f}")
print(f"Binary Cross-Entropy:{bce:.4f}")

# Derivative of MSE loss
def mse_gradient(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

# Derivative of BCE loss
def bce_gradient(y_true, y_pred):
    epsilon = 1e-12  # Small value to avoid division by zero
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / y_true.size

# Calculate gradients
grad_mse = mse_gradient(y_true, y_pred)
grad_bce = bce_gradient(y_true, y_pred)

print("Gradient of MSE Loss:", grad_mse)
print("Gradient of BCE Loss:", grad_bce)

# Define predictions and true labels
predictions = np.linspace(0, 1, 100)
true_lable = 1

# Compute losses
mse_losses = [(true_lable - pred) ** 2 for pred in predictions]
bce_losses = [-true_lable * np.log(max(pred, 1e-12)) - (1 - true_lable) * np.log(max(1 - pred, 1e-12)) for pred in predictions]

# Plot
plt.figure(figsize=(10, 5))
plt.plot(predictions, mse_losses, label='MSE Loss', color='blue')
plt.plot(predictions, bce_losses, label='BCE Loss', color='orange')
plt.xlabel('Predictions')
plt.ylabel('Loss')
plt.title('Loss Functions')
plt.legend()
plt.grid(True)
plt.show()