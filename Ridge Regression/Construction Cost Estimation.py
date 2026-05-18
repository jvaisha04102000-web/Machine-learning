import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

# Dataset
data = {
    'material_cost': [50000, 70000, 90000, 120000, 150000, 180000],
    'labor_cost': [20000, 30000, 35000, 45000, 50000, 60000],
    'project_duration': [3, 4, 5, 6, 8, 10],
    'total_cost': [80000, 110000, 140000, 180000, 220000, 270000]
}

df = pd.DataFrame(data)

# Features & Target
X = df[['material_cost', 'labor_cost', 'project_duration']]
y = df['total_cost']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = Ridge(alpha=2.0)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("R2 Score:", r2_score(y_test, y_pred))

# New Project Prediction
new_project = [[130000, 40000, 7]]
cost = model.predict(new_project)

print("Estimated Construction Cost:", cost[0])