"""
📌 California Housing Price Prediction 🏠
This script loads the California Housing dataset, explores the data, trains a 
Linear Regression model, evaluates performance, and visualizes results.
"""

# ✅ Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ✅ Load Dataset
data = fetch_california_housing(as_frame=True)
df = data.frame  # Convert to Pandas DataFrame

print("✅ First 5 rows of dataset:")
print(df.head())

# ✅ Exploratory Data Analysis (EDA)
print("\n🔍 Dataset Info:")
df.info()

print("\n🛠 Missing Values:")
print(df.isnull().sum())

print("\n📊 Summary Statistics:")
print(df.describe())

# ✅ Data Visualization
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlations")
plt.show()

# ✅ Feature Selection
features = ["MedInc", "AveRooms"]  # Selecting important features
target = "MedHouseVal"
X = df[features]
y = df[target]

# ✅ Train-Test Split (80% Train / 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# ✅ Predictions
y_pred = model.predict(X_test)

# ✅ Model Evaluation
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\n📊 Model Performance:")
print(f"✅ R² Score: {r2:.2f}")
print(f"✅ Mean Absolute Error (MAE): {mae:.2f}")
print(f"✅ Mean Squared Error (MSE): {mse:.2f}")
print(f"✅ Root Mean Squared Error (RMSE): {rmse:.2f}")

# ✅ Visualization: Actual vs Predicted
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.xlabel("Actual House Value ($100,000s)")
plt.ylabel("Predicted House Value ($100,000s)")
plt.title("Actual vs Predicted House Values")
plt.show()
