# Financial Market Trend Prediction using Elastic Net

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Sample Dataset
np.random.seed(42)

data = pd.DataFrame({
    'Open': np.random.uniform(100, 500, 500),
    'High': np.random.uniform(100, 500, 500),
    'Low': np.random.uniform(100, 500, 500),
    'Volume': np.random.uniform(1000, 10000, 500),
    'Market_Sentiment': np.random.uniform(0, 1, 500),
    'Close': np.random.uniform(100, 500, 500)
})

# Features and Target
X = data.drop('Close', axis=1)
y = data['Close']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Elastic Net Model
model = ElasticNet(alpha=0.5, l1_ratio=0.5)

# Training
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# Feature Importance
importance = pd.Series(model.coef_, index=X.columns)

importance.plot(kind='bar')
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Coefficient")
plt.show()