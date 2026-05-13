# Insurance Claim Approval System

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Sample Dataset
data = {
    'ClaimAmount': [5000, 15000, 2000, 25000, 7000, 12000, 3000, 18000],
    'AccidentHistory': [1, 3, 0, 5, 2, 4, 1, 3],
    'PolicyDuration': [5, 2, 6, 1, 4, 3, 7, 2],
    'FraudHistory': [0, 1, 0, 1, 0, 1, 0, 1],
    'Approved': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No']
}

df = pd.DataFrame(data)

# Input and Output
X = df[['ClaimAmount', 'AccidentHistory', 'PolicyDuration', 'FraudHistory']]
y = df['Approved']

# Split Data
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

# Visualization
plt.figure(figsize=(10,6))
plot_tree(model, feature_names=X.columns, class_names=model.classes_, filled=True)
plt.show()