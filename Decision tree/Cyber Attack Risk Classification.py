# Cyber Attack Risk Classification

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Dataset
data = {
    'LoginAttempts': [2, 15, 1, 20, 3, 18, 2, 25],
    'SuspiciousIP': [0, 1, 0, 1, 0, 1, 0, 1],
    'NetworkTraffic': [100, 900, 120, 1000, 150, 850, 130, 1100],
    'Threat': ['Low', 'High', 'Low', 'High', 'Low', 'High', 'Low', 'High']
}

df = pd.DataFrame(data)

# Features and Target
X = df[['LoginAttempts', 'SuspiciousIP', 'NetworkTraffic']]
y = df['Threat']

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

# Security Alerts
for i in range(len(y_pred)):
    print("Threat Level:", y_pred[i])

    if y_pred[i] == 'High':
        print("Security Alert: Possible Cyber Attack Detected")