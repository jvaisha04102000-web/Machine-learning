# Smart Scholarship Eligibility System

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Dataset
data = {
    'FamilyIncome': [20000, 80000, 30000, 100000, 25000, 60000, 15000, 90000],
    'Attendance': [90, 70, 85, 60, 95, 75, 88, 65],
    'Marks': [85, 60, 78, 55, 92, 68, 80, 58],
    'Community': [1, 0, 1, 0, 1, 0, 1, 0],
    'Eligible': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No']
}

df = pd.DataFrame(data)

# Features and Target
X = df[['FamilyIncome', 'Attendance', 'Marks', 'Community']]
y = df['Eligible']

# Train-Test Split
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

# Government Report
report = X_test.copy()
report['PredictedEligibility'] = y_pred

print("\nGovernment Scholarship Report")
print(report)