import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim

# Example data (y = 2x + 1 + noise)
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Add bias term (X0 = 1)
X_b = np.c_[np.ones((100, 1)), X]  # shape (100, 2)

# Loss function: MSE
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Gradient Descent
def gradient_descent(X, y, lr=0.1, n_iters=1000):
    m = len(y)  # số mẫu
    theta = np.random.randn(2, 1)  # random (w, b)
    losses = []

    for i in range(n_iters):
        gradients = 2/m * X.T.dot(X.dot(theta) - y)  # derivative MSE
        theta = theta - lr * gradients  # update parameters
        loss = mse_loss(y, X.dot(theta))
        losses.append(loss)
        
        if i % 100 == 0:
            print(f"Iter {i}: Loss={loss:.4f}, Weights={theta.ravel()}")

    return theta, losses

# Execute Gradient Descent
theta, losses = gradient_descent(X_b, y, lr=0.1, n_iters=1000)
print("Final parameters:", theta.ravel())

# Plot loss reduction
plt.plot(losses)
plt.xlabel("Iteration")
plt.ylabel("MSE Loss")
plt.title("Loss Reduction Over Time")
plt.show()

# Plot data + fitted line
plt.scatter(X, y, alpha=0.6)
plt.plot(X, X_b.dot(theta), color="red", linewidth=2, label="Fitted Line")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()


# TensorFlow version
class LinearModelTF(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.w = tf.Variable(tf.random.normal((1, 1)))
        self.b = tf.Variable(tf.random.normal((1,)))

    def call(self, X):
        return tf.matmul(X, self.w) + self.b

# Training
model_tf = LinearModelTF()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
loss_fn = tf.keras.losses.MeanSquaredError()

losses_tf = []
for epoch in range(100):
    with tf.GradientTape() as tape:
        y_pred = model_tf(X.astype(np.float32))
        loss = loss_fn(y.astype(np.float32), y_pred)
    grads = tape.gradient(loss, [model_tf.w, model_tf.b])
    optimizer.apply_gradients(zip(grads, [model_tf.w, model_tf.b]))
    losses_tf.append(loss.numpy())
    if epoch % 20 == 0:
        print(f"[TensorFlow] Epoch {epoch}, Loss={loss.numpy():.4f}, w={model_tf.w.numpy().ravel()}, b={model_tf.b.numpy()}")

# PyTorch version
class LinearModelTorch(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)  # y = Wx + b

    def forward(self, X):
        return self.linear(X)

# Convert data to tensor
X_torch = torch.tensor(X, dtype=torch.float32)
y_torch = torch.tensor(y, dtype=torch.float32)

model_torch = LinearModelTorch()
optimizer_torch = optim.SGD(model_torch.parameters(), lr=0.1)
loss_fn_torch = nn.MSELoss()

losses_torch = []
for epoch in range(100):
    y_pred = model_torch(X_torch)
    loss = loss_fn_torch(y_pred, y_torch)

    optimizer_torch.zero_grad()
    loss.backward()
    optimizer_torch.step()

    losses_torch.append(loss.item())
    if epoch % 20 == 0:
        w, b = model_torch.linear.weight.item(), model_torch.linear.bias.item()
        print(f"[PyTorch] Epoch {epoch}, Loss={loss.item():.4f}, w={w:.4f}, b={b:.4f}")

# Result comparison
plt.figure(figsize=(12, 5))

# TensorFlow loss
plt.subplot(1, 2, 1)
plt.plot(losses_tf, label="TensorFlow Loss")
plt.title("TensorFlow Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend()

# PyTorch loss
plt.subplot(1, 2, 2)
plt.plot(losses_torch, label="PyTorch Loss", color="orange")
plt.title("PyTorch Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend()

plt.show()

# Plot fitted lines
plt.scatter(X, y, alpha=0.6, label="Data")
plt.plot(X, model_tf(X.astype(np.float32)), "r-", label="TensorFlow Fit")
plt.plot(X, model_torch(X_torch).detach().numpy(), "g--", label="PyTorch Fit")
plt.legend()
plt.title("Linear Regression Fit (TensorFlow vs PyTorch)")
plt.show()
