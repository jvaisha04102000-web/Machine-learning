import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Dataset
data = {
    "skill_assessment": [8,5,9,4,7,
                         3,8,5,9,4],

    "interest_survey": [9,4,8,3,7,
                        2,9,5,8,3],

    "academic_performance": [85,60,90,55,80,
                             50,88,65,92,58],

    "programming_knowledge": [8,3,9,2,7,
                              1,8,4,9,2],

    "communication_skills": [7,5,8,4,6,
                             3,7,5,9,4],

    # 1 = Data Science
    # 2 = Web Development
    # 3 = Cyber Security
    "recommended_course": [1,2,1,3,2,
                           3,1,2,1,3]
}

# Convert dataframe
df = pd.DataFrame(data)

# Features
X = df[[
    "skill_assessment",
    "interest_survey",
    "academic_performance",
    "programming_knowledge",
    "communication_skills"
]]

# Target
y = df["recommended_course"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# KNN Model
model = KNeighborsClassifier(n_neighbors=3)

# Train model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nCOLLEGE COURSE RECOMMENDATION SYSTEM\n")

print("Accuracy Score :", round(accuracy,2))

# User Input
print("\nENTER STUDENT DETAILS\n")

skill_assessment = float(
    input("Skill Assessment Score (1-10) : ")
)

interest_survey = float(
    input("Interest Survey Score (1-10) : ")
)

academic_performance = float(
    input("Academic Performance Percentage : ")
)

programming_knowledge = float(
    input("Programming Knowledge (1-10) : ")
)

communication_skills = float(
    input("Communication Skills (1-10) : ")
)

# Predict
prediction = model.predict([[
    skill_assessment,
    interest_survey,
    academic_performance,
    programming_knowledge,
    communication_skills
]])

# Output
print("\nCOURSE RECOMMENDATION RESULT\n")

if prediction[0] == 1:
    print("Recommended Course : DATA SCIENCE")

elif prediction[0] == 2:
    print("Recommended Course : WEB DEVELOPMENT")

else:
    print("Recommended Course : CYBER SECURITY")

# Matching message
print("Similar student profiles matched successfully")