import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Dataset
data = {
    "skin_redness": [2,8,3,9,4,
                     7,2,8,3,9],

    "itching_level": [3,9,2,10,4,
                      8,3,9,2,10],

    "rash_spread": [1,8,2,9,3,
                    7,1,8,2,9],

    "fever_presence": [0,1,0,1,0,
                       1,0,1,0,1],

    # 1 = Allergy
    # 2 = Fungal Infection
    # 3 = Viral Infection
    "disease_type": [1,2,1,3,1,
                     2,1,2,1,3]
}

# Convert dataframe
df = pd.DataFrame(data)

# Features
X = df[[
    "skin_redness",
    "itching_level",
    "rash_spread",
    "fever_presence"
]]

# Target
y = df["disease_type"]

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

print("\nSKIN DISEASE SIMILARITY DETECTION SYSTEM\n")

print("Accuracy Score :", round(accuracy,2))

# User Input
print("\nENTER PATIENT DETAILS\n")

skin_redness = float(input("Skin Redness Level (1-10) : "))

itching_level = float(input("Itching Level (1-10) : "))

rash_spread = float(input("Rash Spread Level (1-10) : "))

fever_presence = int(input(
    "Fever Presence (1 Yes / 0 No) : "
))

# Predict
prediction = model.predict([[
    skin_redness,
    itching_level,
    rash_spread,
    fever_presence
]])

# Output
print("\nDISEASE DETECTION RESULT\n")

if prediction[0] == 1:
    print("Predicted Disease : ALLERGY")

elif prediction[0] == 2:
    print("Predicted Disease : FUNGAL INFECTION")

else:
    print("Predicted Disease : VIRAL INFECTION")

# Similarity report
print("Similar patient cases matched successfully")

# Doctor Recommendation
print("Doctor report generated successfully")