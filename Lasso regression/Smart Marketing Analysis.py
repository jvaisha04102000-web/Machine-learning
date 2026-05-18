import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

np.random.seed(10)

data = pd.DataFrame({
    'AdBudget': np.random.randint(1000, 10000, 100),
    'SocialMediaClicks': np.random.randint(50, 500, 100),
    'EmailOpenRate': np.random.randint(10, 100, 100),
    'WebsiteVisits': np.random.randint(100, 5000, 100),
    'PreviousPurchases': np.random.randint(0, 20, 100),
    'ConversionRate': np.random.randint(0, 2, 100)
})

X = data.drop('ConversionRate', axis=1)
y = data['ConversionRate']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Model
model = Lasso(alpha=0.01)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)
y_pred_binary = [1 if i > 0.5 else 0 for i in y_pred]

print("Accuracy:", accuracy_score(y_test, y_pred_binary))

# Feature Selection
importance = pd.Series(model.coef_, index=X.columns)

print("\nSelected Important Features:")
print(importance)

# Graph
importance.plot(kind='bar')
plt.title("Marketing Feature Importance")
plt.show()