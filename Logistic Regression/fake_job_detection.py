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
    "salary": [30000, 5000, 45000, 7000, 60000,
               8000, 55000, 4000, 75000, 65000],

    "company_verified": [1,0,1,0,1,
                         0,1,0,1,1],

    "experience_required": [2,0,3,0,5,
                            0,4,0,6,5],

    "description_length": [500,100,650,80,700,
                           120,600,90,750,680],

    "remote_option": [1,1,0,1,0,
                      1,0,1,0,0],

    # 1 = Real Job
    # 0 = Fake Job
    "job_status": [1,0,1,0,1,
                   0,1,0,1,1]
}

# Convert dataframe
df = pd.DataFrame(data)

# Features
X = df[[
    "salary",
    "company_verified",
    "experience_required",
    "description_length",
    "remote_option"
]]

# Target
y = df["job_status"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Logistic Regression model
model = LogisticRegression()

# Train model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nFAKE JOB POSTING DETECTION SYSTEM\n")

print("Accuracy Score :", round(accuracy,2))

print("\nConfusion Matrix\n")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

# User Input
print("\nENTER JOB DETAILS\n")

salary = float(input("Salary Offered : "))

company_verified = int(
    input("Company Verified (1 Yes / 0 No) : ")
)

experience_required = float(
    input("Experience Required (Years) : ")
)

description_length = float(
    input("Job Description Length : ")
)

remote_option = int(
    input("Remote Option (1 Yes / 0 No) : ")
)

# Predict
prediction = model.predict([[
    salary,
    company_verified,
    experience_required,
    description_length,
    remote_option
]])

# Probability
probability = model.predict_proba([[
    salary,
    company_verified,
    experience_required,
    description_length,
    remote_option
]])

# Output
if prediction[0] == 1:
    print("\nPrediction : REAL JOB")
else:
    print("\nPrediction : FAKE JOB")

print(
    "Probability Score :",
    round(np.max(probability) * 100, 2),
    "%"
)