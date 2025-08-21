from sklearn.datasets import load_diabetes
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
import numpy as np

data = load_diabetes(as_frame=True)
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
print(df.head())
print(df.info())

# Calculate correlation matrix
corr = df.corr()

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Matrix")
plt.show()

# Select feature with high correlation
high_corr = corr['target'].abs().sort_values(ascending=False)
print("Features with high correlation to target:")
print(high_corr[high_corr > 0.005])

# Separate the feature and target
X = df[high_corr[high_corr > 0.005].index.drop('target')]
y = df['target']

# Calculate mutual information
mutual_info = mutual_info_regression(X, y)
mutual_info = pd.Series(mutual_info, index=X.columns)
mutual_info = mutual_info.sort_values(ascending=False)
print("Features ranked by mutual information with target:")
print(mutual_info)

# Create a Dataframe for better visualization
feature_importance = pd.DataFrame({
    'Feature': mutual_info.index,
    'Mutual Information': mutual_info.values
})

# Plotting
plt.figure(figsize=(12, 6))
sns.barplot(x='Mutual Information', y='Feature', data=feature_importance)
plt.title("Feature Importance based on Mutual Information")
plt.show()

# Train a Random Forest model
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# Get feature importance
feature_importance = model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importance
})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plotting
plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.gca().invert_yaxis()
plt.title("Feature Importance based on Random Forest")
plt.show()
