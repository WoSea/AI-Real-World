# pip install xgboost lightgbm catboost
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# XGBoost
from xgboost import XGBClassifier
# LightGBM
from lightgbm import LGBMClassifier
# CatBoost
from catboost import CatBoostClassifier

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train XGBoost
xgb_model = XGBClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, use_label_encoder=False, eval_metric="mlogloss"
)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))

# Train LightGBM
lgbm_model = LGBMClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=-1, random_state=42
)
lgbm_model.fit(X_train, y_train)
y_pred_lgbm = lgbm_model.predict(X_test)
print("LightGBM Accuracy:", accuracy_score(y_test, y_pred_lgbm))

# Train CatBoost (silent mode)
cat_model = CatBoostClassifier(
    iterations=100, learning_rate=0.1, depth=3, random_state=42, verbose=0
)
cat_model.fit(X_train, y_train)
y_pred_cat = cat_model.predict(X_test)
print("CatBoost Accuracy:", accuracy_score(y_test, y_pred_cat))
