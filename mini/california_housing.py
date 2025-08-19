import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = fetch_california_housing(as_frame=True)

df = data.frame

# Define features and target
X = df[['MedInc','HouseAge','AveRooms']]
y = df['MedHouseVal']

print(df.info())
print(df.describe())

sns.pairplot(df, vars=['MedInc', 'HouseAge', 'AveRooms', 'MedHouseVal'])

print("Missing values: \n", df.isnull().sum())

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")