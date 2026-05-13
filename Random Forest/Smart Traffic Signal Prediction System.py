# Smart Traffic Signal Prediction System

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Dataset
data = {
    'VehicleCount': [50, 200, 80, 300, 120, 250, 60, 350],
    'Weather': [0, 1, 0, 1, 0, 1, 0, 1],   # 0 = Clear, 1 = Rain
    'PeakHour': [0, 1, 0, 1, 1, 1, 0, 1],
    'Congestion': ['Low', 'High', 'Low', 'High', 'Medium', 'High', 'Low', 'High']
}

df = pd.DataFrame(data)

# Features and Target
X = df[['VehicleCount', 'Weather', 'PeakHour']]
y = df['Congestion']

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

# Smart Signal Timing Recommendation
for prediction in y_pred:
    if prediction == 'High':
        print("Signal Timing: 120 Seconds")
    elif prediction == 'Medium':
        print("Signal Timing: 80 Seconds")
    else:
        print("Signal Timing: 40 Seconds")