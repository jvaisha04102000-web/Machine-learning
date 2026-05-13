# Crop Disease Detection System

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Dataset
data = {
    'SoilMoisture': [30, 80, 40, 90, 35, 85, 45, 95],
    'Temperature': [25, 35, 27, 40, 26, 38, 28, 42],
    'Humidity': [50, 90, 55, 95, 52, 88, 58, 97],
    'Disease': ['No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes']
}

df = pd.DataFrame(data)

# Features and Target
X = df[['SoilMoisture', 'Temperature', 'Humidity']]
y = df['Disease']

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

# Farmer Recommendation Module
for prediction in y_pred:
    if prediction == 'Yes':
        print("Recommendation: Apply Pesticide Immediately")
    else:
        print("Crop Condition is Healthy")