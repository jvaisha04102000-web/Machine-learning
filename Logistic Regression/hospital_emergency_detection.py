import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Dataset
data = {
    "heart_rate": [72,120,80,130,76,
                   140,78,125,85,135],

    "blood_pressure": [120,180,125,190,118,
                       200,122,185,130,195],

    "oxygen_level": [98,82,97,78,99,
                     75,96,80,95,77],

    "temperature": [98.6,103,99,104,98,
                    105,99.2,103.5,100,104.5],

    "breathing_rate": [16,30,18,32,17,
                       35,19,31,20,34],

    # 1 = Emergency
    # 0 = Normal
    "emergency_status": [0,1,0,1,0,
                         1,0,1,0,1]
}

# Convert dataframe
df = pd.DataFrame(data)

# Features
X = df[[
    "heart_rate",
    "blood_pressure",
    "oxygen_level",
    "temperature",
    "breathing_rate"
]]

# Target
y = df["emergency_status"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Logistic Regression Model
model = LogisticRegression()

# Train model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nHOSPITAL EMERGENCY RISK DETECTION SYSTEM\n")

print("Accuracy Score :", round(accuracy,2))

print("\nConfusion Matrix\n")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

# User Input
print("\nENTER PATIENT DETAILS\n")

heart_rate = float(input("Heart Rate : "))

blood_pressure = float(input("Blood Pressure : "))

oxygen_level = float(input("Oxygen Level : "))

temperature = float(input("Body Temperature : "))

breathing_rate = float(input("Breathing Rate : "))

# Predict
prediction = model.predict([[
    heart_rate,
    blood_pressure,
    oxygen_level,
    temperature,
    breathing_rate
]])

# Probability
probability = model.predict_proba([[
    heart_rate,
    blood_pressure,
    oxygen_level,
    temperature,
    breathing_rate
]])

# Output
if prediction[0] == 1:
    print("\nPrediction : EMERGENCY PATIENT")
else:
    print("\nPrediction : NORMAL PATIENT")

print(
    "Risk Probability :",
    round(np.max(probability) * 100, 2),
    "%"
)

# Doctor Alert
if prediction[0] == 1:
    print("DOCTOR ALERT : Immediate medical attention required")