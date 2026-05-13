# Restaurant Food Demand Prediction

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Dataset
data = {
    'SeasonSales': [200, 500, 150, 700, 300, 650, 250, 720],
    'FestivalDay': [0, 1, 0, 1, 0, 1, 0, 1],
    'CustomerTraffic': [50, 150, 40, 180, 70, 160, 60, 190],
    'Demand': ['Low', 'High', 'Low', 'High', 'Medium', 'High', 'Medium', 'High']
}

df = pd.DataFrame(data)

# Features and Target
X = df[['SeasonSales', 'FestivalDay', 'CustomerTraffic']]
y = df['Demand']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Inventory Dashboard
dashboard = X_test.copy()
dashboard['PredictedDemand'] = y_pred

print("\nInventory Planning Dashboard")
print(dashboard)