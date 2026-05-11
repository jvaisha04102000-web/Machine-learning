import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Load dataset
df = pd.read_csv("cab_fare_dataset.csv")

# Features
X = df[[
    "distance",
    "traffic",
    "time_hours",
    "weather",
    "peak_hour"
]]

# Target
y = df["fare"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy metrics
mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

r2 = r2_score(y_test, y_pred)

print("\nCAB FARE MODEL ACCURACY\n")

print("MAE :", mae)
print("RMSE :", rmse)
print("R2 Score :", r2)

# Save model
joblib.dump(model, "model.pkl")

print("\nModel Saved Successfully")