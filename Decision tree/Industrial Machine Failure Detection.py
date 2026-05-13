# Industrial Machine Failure Detection

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Dataset
data = {
    'Temperature': [70, 95, 65, 110, 80, 100, 72, 105],
    'Vibration': [20, 80, 15, 90, 30, 85, 25, 88],
    'SensorValue': [40, 95, 35, 100, 45, 98, 38, 96],
    'Failure': ['No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes']
}

df = pd.DataFrame(data)

# Features and Target
X = df[['Temperature', 'Vibration', 'SensorValue']]
y = df['Failure']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Maintenance Alert
for i in range(len(y_pred)):
    if y_pred[i] == 'Yes':
        print("Maintenance Alert: Machine Failure Predicted")
    else:
        print("Machine Running Normally")