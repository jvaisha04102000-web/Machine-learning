import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Sample Dataset
np.random.seed(42)

data = pd.DataFrame({
    'WorkingHours': np.random.randint(6, 12, 100),
    'Experience': np.random.randint(1, 15, 100),
    'ProjectsCompleted': np.random.randint(1, 20, 100),
    'TrainingHours': np.random.randint(5, 50, 100),
    'LateArrivals': np.random.randint(0, 10, 100),
    'ProductivityScore': np.random.randint(50, 100, 100)
})

X = data.drop('ProductivityScore', axis=1)
y = data['ProductivityScore']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Lasso Model
model = Lasso(alpha=0.1)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# Feature Importance
importance = pd.Series(model.coef_, index=X.columns)

print("\nFeature Importance:")
print(importance)

# Graph
importance.plot(kind='bar')
plt.title("Employee Feature Importance")
plt.xlabel("Features")
plt.ylabel("Coefficient Value")
plt.show()