import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Dataset
data = {
    'travel_season': [1, 2, 3, 1, 2, 3, 1, 3],
    'flight_distance': [500, 800, 1200, 1500, 2000, 2500, 3000, 3500],
    'airline_category': [1, 2, 1, 3, 2, 3, 1, 2],
    'ticket_price': [4000, 6000, 7500, 9000, 11000, 13000, 15000, 17000]
}

df = pd.DataFrame(data)

# Features & Target
X = df[['travel_season', 'flight_distance', 'airline_category']]
y = df['ticket_price']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Model
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
mse = mean_squared_error(y_test, y_pred)

print("Mean Squared Error:", mse)

# New Flight Prediction
new_flight = [[2, 1800, 2]]
price = model.predict(new_flight)

print("Predicted Ticket Price:", price[0])