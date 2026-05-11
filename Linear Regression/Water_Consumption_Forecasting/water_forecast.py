import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Dataset
data = {
    "occupancy": [10,15,8,20,12,18,6,25,14,16],
    "temperature": [30,32,28,35,31,34,27,36,30,33],
    "season": [1,2,1,3,2,3,1,3,2,2],
    "monthly_usage": [1500,1800,1200,2500,1600,2200,1000,3000,1700,2000],
    "water_consumption": [1600,1900,1250,2700,1700,2400,1100,3300,1800,2100]
}

df = pd.DataFrame(data)

# Features
X = df[[
    "occupancy",
    "temperature",
    "season",
    "monthly_usage"
]]

# Target
y = df["water_consumption"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = LinearRegression()

# Train
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

r2 = r2_score(y_test, y_pred)

print("\nWATER CONSUMPTION FORECASTING SYSTEM\n")

print("MAE :", round(mae,2))
print("RMSE :", round(rmse,2))
print("R2 Score :", round(r2,2))

# User input
print("\nENTER DETAILS\n")

occupancy = float(input("Occupancy Count : "))
temperature = float(input("Temperature : "))
season = float(input("Season (1-3) : "))
monthly_usage = float(input("Monthly Usage : "))

# Prediction
prediction = model.predict([[
    occupancy,
    temperature,
    season,
    monthly_usage
]])

predicted_value = round(prediction[0],2)

print("\nPredicted Water Consumption :", predicted_value, "Liters")

# Alert
if predicted_value > 2500:
    print("ALERT : Water usage exceeds limit")
else:
    print("Water usage is under control")