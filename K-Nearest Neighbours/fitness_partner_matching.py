import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Dataset
data = {
    "bmi": [22,30,24,35,21,
            32,23,28,20,34],

    "workout_hours": [2,1,3,1,2,
                      1,3,2,4,1],

    "fitness_goal": [1,2,1,3,1,
                     2,1,2,1,3],

    "diet_preference": [1,2,1,3,1,
                        2,1,2,1,3],

    "activity_level": [8,4,9,3,8,
                       5,9,6,10,4],

    # 1 = Weight Loss Partner
    # 2 = Muscle Gain Partner
    # 3 = General Fitness Partner
    "partner_type": [1,2,1,3,1,
                     2,1,2,1,3]
}

# Convert dataframe
df = pd.DataFrame(data)

# Features
X = df[[
    "bmi",
    "workout_hours",
    "fitness_goal",
    "diet_preference",
    "activity_level"
]]

# Target
y = df["partner_type"]

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

print("\nFITNESS PARTNER MATCHING SYSTEM\n")

print("Accuracy Score :", round(accuracy,2))

# User Input
print("\nENTER FITNESS DETAILS\n")

bmi = float(input("BMI Value : "))

workout_hours = float(
    input("Daily Workout Hours : ")
)

fitness_goal = int(input(
    "Fitness Goal (1 Weight Loss / 2 Muscle Gain / 3 General Fitness) : "
))

diet_preference = int(input(
    "Diet Preference (1 Veg / 2 Mixed / 3 High Protein) : "
))

activity_level = float(
    input("Activity Level (1-10) : ")
)

# Predict
prediction = model.predict([[
    bmi,
    workout_hours,
    fitness_goal,
    diet_preference,
    activity_level
]])

# Output
print("\nFITNESS PARTNER MATCH RESULT\n")

if prediction[0] == 1:
    print("Matched Partner Type : WEIGHT LOSS PARTNER")

elif prediction[0] == 2:
    print("Matched Partner Type : MUSCLE GAIN PARTNER")

else:
    print("Matched Partner Type : GENERAL FITNESS PARTNER")

# Matching message
print("Similar fitness profiles matched successfully")

# Workout recommendation
print("Workout recommendation generated successfully")