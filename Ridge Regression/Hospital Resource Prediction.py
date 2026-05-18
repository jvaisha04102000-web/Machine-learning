import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

# Sample Dataset
data = {
    'bed_occupancy': [120, 150, 180, 200, 220, 250, 270, 300],
    'doctor_availability': [20, 25, 30, 32, 35, 40, 42, 45],
    'emergency_cases': [15, 18, 25, 30, 35, 40, 42, 50],
    'resources_needed': [100, 120, 150, 170, 190, 220, 240, 270]
}

df = pd.DataFrame(data)

# Features and Target
X = df[['bed_occupancy', 'doctor_availability', 'emergency_cases']]
y = df['resources_needed']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Ridge Regression Model
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# New Prediction
new_data = [[230, 36, 38]]
prediction = model.predict(new_data)

print("Predicted Resources Needed:", prediction[0])