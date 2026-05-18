import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

np.random.seed(5)

data = pd.DataFrame({
    'Age': np.random.randint(20, 80, 100),
    'BloodPressure': np.random.randint(80, 180, 100),
    'SugarLevel': np.random.randint(70, 250, 100),
    'Cholesterol': np.random.randint(100, 300, 100),
    'BMI': np.random.uniform(18, 40, 100),
    'DiseaseRisk': np.random.randint(0, 100, 100)
})

X = data.drop('DiseaseRisk', axis=1)
y = data['DiseaseRisk']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Model
model = Lasso(alpha=0.5)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print("R2 Score:", r2_score(y_test, y_pred))

# Important Risk Factors
importance = pd.Series(model.coef_, index=X.columns)

print("\nImportant Medical Risk Factors:")
print(importance)

# Graph
importance.plot(kind='bar')
plt.title("Medical Risk Feature Importance")
plt.show()