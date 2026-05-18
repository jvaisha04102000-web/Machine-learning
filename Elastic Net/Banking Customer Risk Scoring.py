# Banking Customer Risk Scoring using Elastic Net

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Dataset
np.random.seed(7)

data = pd.DataFrame({
    'Income': np.random.uniform(20000, 150000, 500),
    'Loan_Amount': np.random.uniform(5000, 100000, 500),
    'Credit_Score': np.random.uniform(300, 900, 500),
    'EMI': np.random.uniform(1000, 50000, 500),
    'Existing_Loans': np.random.randint(0, 5, 500),
    'Risk_Score': np.random.uniform(0, 100, 500)
})

X = data.drop('Risk_Score', axis=1)
y = data['Risk_Score']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Model
model = ElasticNet(alpha=0.6, l1_ratio=0.5)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("Risk Prediction Accuracy:", r2_score(y_test, y_pred))

# Feature Selection
importance = pd.Series(model.coef_, index=X.columns)

print("\nImportant Financial Indicators:")
print(importance)