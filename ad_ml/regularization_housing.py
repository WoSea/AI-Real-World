from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np

# Load California housing dataset
data = fetch_california_housing()
X, y = data.data, data.target
print("Feature names:", data.feature_names)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)

# Ridge Regression
ridge = Ridge(alpha=1.0)   # alpha = strength of regularization
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)

# Lasso Regression
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)

# Print results
print("Linear Regression MSE:", mse_lr)
print("Ridge Regression MSE:", mse_ridge)
print("Lasso Regression MSE:", mse_lasso)

print(f"Coefficient values (Ridge): {ridge.coef_}")
print(f"Coefficient values (Lasso): {lasso.coef_}")
# Show number of features used in Lasso (feature selection effect)
print("Number of features used in Lasso:", np.sum(lasso.coef_ != 0))