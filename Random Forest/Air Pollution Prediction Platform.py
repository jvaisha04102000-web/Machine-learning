# Air Pollution Prediction Platform

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Dataset
data = {
    'AQI': [50, 200, 70, 300, 90, 250, 60, 350],
    'VehicleDensity': [100, 500, 150, 700, 200, 650, 120, 800],
    'WeatherCondition': [0, 1, 0, 1, 0, 1, 0, 1],
    'PollutionLevel': ['Low', 'High', 'Low', 'High', 'Medium', 'High', 'Low', 'High']
}

df = pd.DataFrame(data)

# Features and Target
X = df[['AQI', 'VehicleDensity', 'WeatherCondition']]
y = df['PollutionLevel']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# City-wise Analytics Dashboard
pollution_counts = pd.Series(y_pred).value_counts()

plt.bar(pollution_counts.index, pollution_counts.values)
plt.xlabel("Pollution Level")
plt.ylabel("Count")
plt.title("Air Pollution Analytics Dashboard")
plt.show()