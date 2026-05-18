import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

np.random.seed(1)

data = pd.DataFrame({
    'Area': np.random.randint(800, 4000, 100),
    'Bedrooms': np.random.randint(1, 6, 100),
    'Bathrooms': np.random.randint(1, 5, 100),
    'Parking': np.random.randint(0, 3, 100),
    'Age': np.random.randint(1, 20, 100),
    'Price': np.random.randint(2000000, 15000000, 100)
})

X = data.drop('Price', axis=1)
y = data['Price']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Model
model = Lasso(alpha=1000)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print("R2 Score:", r2_score(y_test, y_pred))

# Important Features
importance = pd.Series(model.coef_, index=X.columns)

print("\nImportant Factors:")
print(importance)

# Visualization
importance.plot(kind='bar')
plt.title("Real Estate Important Features")
plt.show()