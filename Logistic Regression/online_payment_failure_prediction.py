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
    "internet_speed": [50,5,45,3,60,
                       2,55,4,70,6],

    "transaction_amount": [1000,5000,1500,7000,1200,
                           8000,2000,6000,1800,5500],

    "server_response_time": [1,10,2,12,1.5,
                             15,2,11,1,9],

    "device_type": [1,0,1,0,1,
                    0,1,0,1,0],

    "browser_stability": [9,2,8,1,10,
                          2,9,3,10,2],

    # 1 = Success
    # 0 = Failed
    "payment_status": [1,0,1,0,1,
                       0,1,0,1,0]
}

# Convert dataframe
df = pd.DataFrame(data)

# Features
X = df[[
    "internet_speed",
    "transaction_amount",
    "server_response_time",
    "device_type",
    "browser_stability"
]]

# Target
y = df["payment_status"]

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

print("\nONLINE PAYMENT FAILURE PREDICTION SYSTEM\n")

print("Accuracy Score :", round(accuracy,2))

print("\nConfusion Matrix\n")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

# User Input
print("\nENTER PAYMENT DETAILS\n")

internet_speed = float(input("Internet Speed (Mbps) : "))

transaction_amount = float(
    input("Transaction Amount : ")
)

server_response_time = float(
    input("Server Response Time : ")
)

device_type = int(
    input("Device Type (1 Mobile / 0 Desktop) : ")
)

browser_stability = float(
    input("Browser Stability Score (1-10) : ")
)

# Predict
prediction = model.predict([[
    internet_speed,
    transaction_amount,
    server_response_time,
    device_type,
    browser_stability
]])

# Probability
probability = model.predict_proba([[
    internet_speed,
    transaction_amount,
    server_response_time,
    device_type,
    browser_stability
]])

# Output
if prediction[0] == 1:
    print("\nPrediction : PAYMENT SUCCESSFUL")
else:
    print("\nPrediction : PAYMENT FAILURE")

print(
    "Success/Failure Probability :",
    round(np.max(probability) * 100, 2),
    "%"
)

# Risk Alert
if prediction[0] == 0:
    print("RISK ALERT : High payment failure risk detected")