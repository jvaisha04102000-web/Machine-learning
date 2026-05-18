import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

# Dataset
data = {
    'product_sales': [1000, 1500, 2000, 2500, 3000, 3500],
    'marketing_spend': [500, 700, 900, 1200, 1500, 1800],
    'seasonal_index': [1, 2, 2, 3, 3, 4],
    'profit': [10000, 15000, 20000, 26000, 32000, 39000]
}

df = pd.DataFrame(data)

# Features & Target
X = df[['product_sales', 'marketing_spend', 'seasonal_index']]
y = df['profit']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Model
model = Ridge(alpha=1.5)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("MAE:", mean_absolute_error(y_test, y_pred))

# New Retail Prediction
new_data = [[2800, 1400, 3]]
profit = model.predict(new_data)

print("Predicted Profit:", profit[0])