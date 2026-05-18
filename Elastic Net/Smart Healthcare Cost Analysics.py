import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# Sample Dataset
np.random.seed(1)

data = pd.DataFrame({
    'Age': np.random.randint(18, 80, 400),
    'BMI': np.random.uniform(18, 40, 400),
    'Children': np.random.randint(0, 5, 400),
    'Smoking': np.random.randint(0, 2, 400),
    'Hospital_Visits': np.random.randint(1, 20, 400),
    'Medical_Cost': np.random.uniform(2000, 50000, 400)
})

X = data.drop('Medical_Cost', axis=1)
y = data['Medical_Cost']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Model
model = ElasticNet(alpha=0.3, l1_ratio=0.7)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

# Optimized Features
features = pd.Series(model.coef_, index=X.columns)
print("\nFeature Optimization:")
print(features)