# GridSearchCV: Browse the whole => accurate but slow.
# RandomizedSearchCV:  n_iter randomly select => faster, suitable for multiple hyperparameter.
#Optuna (Bayesian Optimization): Learn from trial history => Select Smart more, converge quickly to the best area
# pip install optuna
import optuna
from xgboost import XGBClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
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
# estimator = RandomForestClassifier(random_state=42)

# Define XGBoost model
estimator = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)

# Define parameter grid
# param_grid = {
#     'n_estimators': [10, 50, 100, 200],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 5, 10, 20]
# }
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}

# Grid Search
grid_search = GridSearchCV(estimator, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
print("Grid Search Best Params:", grid_search.best_params_)
print("Grid Search Best Score:", grid_search.best_score_)

# Random Search
# param_dist = {
#     'n_estimators': np.arange(10, 201, 150),
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 5, 10,20]
# }
param_dist = {
    'n_estimators': np.arange(50, 300, 50),
    'max_depth': np.arange(3, 10),
    'learning_rate': np.linspace(0.01, 0.3, 30),
    'subsample': np.linspace(0.5, 1.0, 10)
}

random_search = RandomizedSearchCV(estimator, param_distributions=param_dist, 
                                   n_iter=20, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)
print("Random Search Best Params:", random_search.best_params_)
print("Random Search Best Score:", random_search.best_score_)

# Bayesian Optimization with Optuna
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }
    model = XGBClassifier(**params, use_label_encoder=False, eval_metric="mlogloss", random_state=42)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1).mean()
    return score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30, n_jobs=-1)

print("\nOptuna Best Params:", study.best_params)
print("Optuna Best CV Score:", study.best_value)

# Evaluate best model
best_model_grid = grid_search.best_estimator_
y_pred_grid = best_model_grid.predict(X_test)
print("\nFinal Accuracy (Grid Search best model):", accuracy_score(y_test, y_pred_grid))

best_model_random = random_search.best_estimator_
y_pred_random = best_model_random.predict(X_test)
print("Final Accuracy (Random Search best model):", accuracy_score(y_test, y_pred_random))

best_optuna = XGBClassifier(**study.best_params, use_label_encoder=False, eval_metric="mlogloss", random_state=42)
best_optuna.fit(X_train, y_train)
y_pred_optuna = best_optuna.predict(X_test)
print("Final Accuracy (Optuna best model):", accuracy_score(y_test, y_pred_optuna))