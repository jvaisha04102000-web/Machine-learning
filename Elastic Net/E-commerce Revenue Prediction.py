import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Dataset
np.random.seed(5)

data = pd.DataFrame({
    'Page_Views': np.random.randint(1, 100, 500),
    'Time_Spent': np.random.uniform(1, 30, 500),
    'Cart_Items': np.random.randint(0, 20, 500),
    'Purchases': np.random.randint(0, 10, 500),
    'Ads_Clicked': np.random.randint(0, 15, 500),
    'Revenue': np.random.uniform(100, 10000, 500)
})

X = data.drop('Revenue', axis=1)
y = data['Revenue']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Model
model = ElasticNet(alpha=0.4, l1_ratio=0.6)

# Training
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Result
print("Revenue Prediction R2 Score:", r2_score(y_test, y_pred))

# Feature Importance
importance = pd.Series(model.coef_, index=X.columns)
print("\nUser Behavior Importance:")
print(importance)