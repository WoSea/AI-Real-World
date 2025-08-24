import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train models with different regularizations
# L2 Regularization (Ridge)
model_l2 = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=200, multi_class='multinomial')
model_l2.fit(X_train, y_train)

# L1 Regularization (Lasso)
model_l1 = LogisticRegression(penalty='l1', C=1.0, solver='saga', max_iter=200, multi_class='multinomial')
model_l1.fit(X_train, y_train)

model_en = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, C=1.0, max_iter=200, multi_class='multinomial')
model_en.fit(X_train, y_train)

# Extract coefficients
coef_l2 = model_l2.coef_.ravel()
coef_l1 = model_l1.coef_.ravel()
coef_en = model_en.coef_.ravel()

# Plot comparison
plt.figure(figsize=(12, 6))
indices = np.arange(len(coef_l2))

plt.plot(indices, coef_l2, 'o-', label='L2 (Ridge)')
plt.plot(indices, coef_l1, 's-', label='L1 (Lasso)')
plt.plot(indices, coef_en, 'x-', label='Elastic Net')

plt.xlabel("Coefficient index (flattened across classes/features)")
plt.ylabel("Coefficient value")
plt.title("Comparison of Logistic Regression Coefficients with Regularization")
plt.legend()
plt.grid(True)
plt.show()