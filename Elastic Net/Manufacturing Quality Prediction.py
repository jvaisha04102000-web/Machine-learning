# Manufacturing Quality Prediction using Elastic Net

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Dataset
np.random.seed(10)

data = pd.DataFrame({
    'Temperature': np.random.uniform(20, 100, 600),
    'Pressure': np.random.uniform(1, 10, 600),
    'Humidity': np.random.uniform(20, 80, 600),
    'Machine_Speed': np.random.uniform(100, 500, 600),
    'Vibration': np.random.uniform(0, 5, 600),
    'Quality_Score': np.random.uniform(50, 100, 600)
})

X = data.drop('Quality_Score', axis=1)
y = data['Quality_Score']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Model
model = ElasticNet(alpha=0.2, l1_ratio=0.5)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("MSE:", mean_squared_error(y_test, y_pred))

# Noise Handling / Feature Selection
importance = pd.Series(model.coef_, index=X.columns)
print("\nSensor Feature Importance:")
print(importance)