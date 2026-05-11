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
    "academic_marks": [85,45,78,50,92,
                       40,88,55,95,60],

    "entrance_score": [90,35,80,40,96,
                       30,89,50,98,58],

    "reservation_category": [1,1,0,0,1,
                             0,1,0,1,0],

    "interview_marks": [88,40,75,45,94,
                        35,90,50,97,60],

    "attendance_percentage": [95,60,85,65,98,
                              55,92,70,99,75],

    # 1 = Eligible
    # 0 = Not Eligible
    "admission_status": [1,0,1,0,1,
                         0,1,0,1,0]
}

# Convert dataframe
df = pd.DataFrame(data)

# Features
X = df[[
    "academic_marks",
    "entrance_score",
    "reservation_category",
    "interview_marks",
    "attendance_percentage"
]]

# Target
y = df["admission_status"]

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

print("\nUNIVERSITY ADMISSION ELIGIBILITY PREDICTOR\n")

print("Accuracy Score :", round(accuracy,2))

print("\nConfusion Matrix\n")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

# User Input
print("\nENTER STUDENT DETAILS\n")

academic_marks = float(input("Academic Marks : "))

entrance_score = float(input("Entrance Exam Score : "))

reservation_category = int(
    input("Reservation Category (1 Yes / 0 No) : ")
)

interview_marks = float(input("Interview Marks : "))

attendance_percentage = float(
    input("Attendance Percentage : ")
)

# Predict
prediction = model.predict([[
    academic_marks,
    entrance_score,
    reservation_category,
    interview_marks,
    attendance_percentage
]])

# Probability
probability = model.predict_proba([[
    academic_marks,
    entrance_score,
    reservation_category,
    interview_marks,
    attendance_percentage
]])

# Output
if prediction[0] == 1:
    print("\nPrediction : STUDENT IS ELIGIBLE")
else:
    print("\nPrediction : STUDENT IS NOT ELIGIBLE")

print(
    "Eligibility Probability :",
    round(np.max(probability) * 100, 2),
    "%"
)

# Merit Analysis
if prediction[0] == 1 and academic_marks > 90:
    print("MERIT STATUS : High Merit Candidate")

elif prediction[0] == 1:
    print("MERIT STATUS : Eligible Candidate")