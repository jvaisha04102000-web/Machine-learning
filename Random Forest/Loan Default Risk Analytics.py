# Loan Default Risk Analytics

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Dataset
data = {
    'CreditScore': [750, 500, 680, 450, 720, 400, 800, 520],
    'Salary': [60000, 25000, 50000, 20000, 70000, 18000, 90000, 30000],
    'EMIHistory': [1, 0, 1, 0, 1, 0, 1, 0],
    'BankTransactions': [200, 50, 180, 40, 250, 30, 300, 60],
    'Risk': ['Low', 'High', 'Low', 'High', 'Low', 'High', 'Low', 'High']
}

df = pd.DataFrame(data)

# Features and Target
X = df[['CreditScore', 'Salary', 'EMIHistory', 'BankTransactions']]
y = df['Risk']

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

# Risk Visualization Dashboard
risk_counts = pd.Series(y_pred).value_counts()

plt.bar(risk_counts.index, risk_counts.values)
plt.xlabel("Risk Level")
plt.ylabel("Count")
plt.title("Loan Default Risk Dashboard")
plt.show()