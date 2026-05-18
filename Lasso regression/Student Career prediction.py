import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

np.random.seed(7)

data = pd.DataFrame({
    'CodingSkill': np.random.randint(1, 10, 100),
    'CommunicationSkill': np.random.randint(1, 10, 100),
    'ProblemSolving': np.random.randint(1, 10, 100),
    'CGPA': np.random.uniform(5, 10, 100),
    'ProjectsDone': np.random.randint(0, 15, 100),
    'CareerScore': np.random.randint(50, 100, 100)
})

X = data.drop('CareerScore', axis=1)
y = data['CareerScore']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Model
model = Lasso(alpha=0.1)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print("R2 Score:", r2_score(y_test, y_pred))

# Important Skills
importance = pd.Series(model.coef_, index=X.columns)

print("\nImportant Skills:")
print(importance)

# Visualization
importance.plot(kind='bar')
plt.title("Student Career Feature Importance")
plt.show()