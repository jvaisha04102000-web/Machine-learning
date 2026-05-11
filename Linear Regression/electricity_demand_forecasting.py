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
    "temperature": [28,30,35,40,25,32,38,29,33,36],

    "historical_consumption": [200,250,400,500,180,300,450,220,350,420],

    "festival_season": [0,0,1,1,0,0,1,0,1,1],

    "hourly_usage": [20,25,40,50,18,30,45,22,35,42],

    "electricity_demand": [220,280,450,580,200,340,520,250,410,490]
}

# Convert dataframe
df = pd.DataFrame(data)

# Features
X = df[[
    "temperature",
    "historical_consumption",
    "festival_season",
    "hourly_usage"
]]

# Target
y = df["electricity_demand"]

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

# Accuracy Metrics
mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

r2 = r2_score(y_test, y_pred)

print("\nELECTRICITY DEMAND FORECASTING SYSTEM\n")

print("MAE :", round(mae,2))
print("RMSE :", round(rmse,2))
print("R2 Score :", round(r2,2))

# User Input
print("\nENTER ELECTRICITY DETAILS\n")

temperature = float(input("Temperature : "))

historical_consumption = float(
    input("Historical Consumption : ")
)

festival_season = float(
    input("Festival Season (0 or 1) : ")
)

hourly_usage = float(input("Hourly Usage : "))

# Predict
prediction = model.predict([[
    temperature,
    historical_consumption,
    festival_season,
    hourly_usage
]])

predicted_demand = round(prediction[0],2)

print(
    "\nPredicted Electricity Demand :",
    predicted_demand,
    "Units"
)

# Analysis
if predicted_demand > 500:
    print("High Electricity Demand Expected")

elif predicted_demand > 300:
    print("Moderate Electricity Demand")

else:
    print("Low Electricity Demand")
