# Retail Inventory Optimization

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Dataset
data = {
    'SalesHistory': [200, 700, 300, 900, 250, 850, 400, 950],
    'FestivalDemand': [0, 1, 0, 1, 0, 1, 0, 1],
    'SeasonTrend': [1, 2, 1, 3, 1, 3, 2, 3],
    'StockRequirement': ['Low', 'High', 'Medium', 'High', 'Low', 'High', 'Medium', 'High']
}

df = pd.DataFrame(data)

# Features and Target
X = df[['SalesHistory', 'FestivalDemand', 'SeasonTrend']]
y = df['StockRequirement']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Automated Reorder Alerts
for prediction in y_pred:
    if prediction == 'High':
        print("Reorder Alert: Increase Inventory")
    elif prediction == 'Medium':
        print("Maintain Moderate Stock")
    else:
        print("Current Inventory is Enough")