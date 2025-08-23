# GridSearchCV: Browse the whole => accurate but slow.
# RandomizedSearchCV:  n_iter randomly select => faster, suitable for multiple hyperparameter.

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Load data
data = load_iris()
X,y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("Train set shape:", X_train.shape, y_train.shape)
print(f"Feature names: {data.feature_names}")
print(f"Class names: {data.target_names}")

# Define model
estimator = RandomForestClassifier(random_state=42)

# Define parameter grid
param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10, 20]
}

# Grid Search
grid_search = GridSearchCV(estimator, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
print("Grid Search Best Params:", grid_search.best_params_)
print("Grid Search Best Score:", grid_search.best_score_)

# Random Search
param_dist = {
    'n_estimators': np.arange(10, 201, 150),
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10,20]
}
random_search = RandomizedSearchCV(estimator, param_distributions=param_dist, 
                                   n_iter=20, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)
print("Random Search Best Params:", random_search.best_params_)
print("Random Search Best Score:", random_search.best_score_)

# Evaluate best model
best_model_grid = grid_search.best_estimator_
y_pred_grid = best_model_grid.predict(X_test)
print("Final Accuracy (Grid Search best model):", accuracy_score(y_test, y_pred_grid))

best_model_random = random_search.best_estimator_
y_pred_random = best_model_random.predict(X_test)
print("Final Accuracy (Random Search best model):", accuracy_score(y_test, y_pred_random))