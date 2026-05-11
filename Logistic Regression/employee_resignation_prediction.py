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
    "attendance": [95,70,88,60,92,
                   55,85,65,98,72],

    "salary": [50000,25000,45000,22000,60000,
               20000,48000,24000,70000,30000],

    "work_pressure": [2,8,3,9,2,
                      10,4,8,1,7],

    "overtime_hours": [5,25,8,30,4,
                       35,10,28,2,20],

    "satisfaction_score": [9,3,8,2,9,
                           1,7,3,10,4],

    # 1 = Stay
    # 0 = Resign
    "employee_status": [1,0,1,0,1,
                        0,1,0,1,0]
}

# Convert dataframe
df = pd.DataFrame(data)

# Features
X = df[[
    "attendance",
    "salary",
    "work_pressure",
    "overtime_hours",
    "satisfaction_score"
]]

# Target
y = df["employee_status"]

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

print("\nEMPLOYEE RESIGNATION PREDICTION SYSTEM\n")

print("Accuracy Score :", round(accuracy,2))

print("\nConfusion Matrix\n")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

# User Input
print("\nENTER EMPLOYEE DETAILS\n")

attendance = float(input("Attendance Percentage : "))

salary = float(input("Salary : "))

work_pressure = float(input("Work Pressure (1-10) : "))

overtime_hours = float(input("Overtime Hours : "))

satisfaction_score = float(
    input("Satisfaction Score (1-10) : ")
)

# Predict
prediction = model.predict([[
    attendance,
    salary,
    work_pressure,
    overtime_hours,
    satisfaction_score
]])

# Probability
probability = model.predict_proba([[
    attendance,
    salary,
    work_pressure,
    overtime_hours,
    satisfaction_score
]])

# Output
if prediction[0] == 1:
    print("\nPrediction : EMPLOYEE WILL STAY")
else:
    print("\nPrediction : EMPLOYEE MAY RESIGN")

print(
    "Probability Score :",
    round(np.max(probability) * 100, 2),
    "%"
)

# HR Alert
if prediction[0] == 0:
    print("HR ALERT : High resignation risk detected")

