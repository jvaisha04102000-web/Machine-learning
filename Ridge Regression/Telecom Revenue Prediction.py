import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Dataset
data = {
    'recharge_amount': [100, 200, 300, 400, 500, 600, 700, 800],
    'call_minutes': [50, 80, 100, 130, 160, 190, 220, 250],
    'internet_usage': [2, 4, 6, 8, 10, 12, 15, 18],
    'monthly_revenue': [150, 280, 350, 480, 600, 720, 850, 950]
}

df = pd.DataFrame(data)

# Features and Target
X = df[['recharge_amount', 'call_minutes', 'internet_usage']]
y = df['monthly_revenue']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Model
model = Ridge(alpha=0.5)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
mse = mean_squared_error(y_test, y_pred)

print("Mean Squared Error:", mse)

# New User Prediction
new_user = [[550, 170, 11]]
revenue = model.predict(new_user)

print("Predicted Revenue:", revenue[0])