import kagglehub
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

path = kagglehub.dataset_download("contactprad/bike-share-daily-data")
print("Path to dataset files:", path)

# Load the dataset
df = pd.read_csv(path +"/bike_sharing_daily.csv")
print(df.head())
print(df.info())

# Convert dteday column to datetime
df['dteday'] = pd.to_datetime(df['dteday'])

# Create new features
df['year'] = df['dteday'].dt.year
df['month'] = df['dteday'].dt.month
df['day_of_week'] = df['dteday'].dt.day_name()

# Display the updated DataFrame
print("Updated DataFrame with new features:")
print(df.head())

# Select feature and target
X = df[['temp', 'hum', 'windspeed', 'year', 'month']]
y = df['cnt']

# Apply polynomial feature transformation
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Display the transformed features
print("Original features:")
print(X[:5])
print("Transformed features:")
print(pd.DataFrame(X_poly[:4], columns=poly.get_feature_names_out()).head())

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_poly_train, X_poly_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Train and evaluate model with original features
model_original = LinearRegression()
model_original.fit(X_train, y_train)
y_pred_original = model_original.predict(X_test)

# Evaluate the model
msa = mean_absolute_error(y_test, y_pred_original)
mse = mean_squared_error(y_test, y_pred_original)
r2 = r2_score(y_test, y_pred_original)
print("Original Features Model Performance:")
print("Mean Absolute Error:", msa)
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)


# Train and evaluate model with polynomial features
model_poly = LinearRegression()
model_poly.fit(X_poly_train, y_train)
y_pred_poly = model_poly.predict(X_poly_test)

# Evaluate the model
msa_poly = mean_absolute_error(y_test, y_pred_poly)
mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)
print("Polynomial Features Model Performance:")
print("Mean Absolute Error:", msa_poly)
print("Mean Squared Error:", mse_poly)
print("R^2 Score:", r2_poly)

# Compare results
print("Comparison of Original and Polynomial Features Model Performance:")
print(f"Original Features - Mean Absolute Error: {msa:.2f}")
print(f"Original Features - Mean Squared Error: {mse:.2f}")
print(f"Original Features - R^2 Score: {r2:.2f}")
print(f"Polynomial Features - Mean Absolute Error: {msa_poly:.2f}")
print(f"Polynomial Features - Mean Squared Error: {mse_poly:.2f}")
print(f"Polynomial Features - R^2 Score: {r2_poly:.2f}")
