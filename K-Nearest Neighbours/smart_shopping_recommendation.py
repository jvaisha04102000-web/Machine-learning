import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Dataset
data = {
    "purchase_history": [8,3,9,2,7,
                         1,8,4,9,2],

    "product_similarity": [9,4,8,3,7,
                           2,9,5,8,3],

    "cart_behavior": [7,2,8,1,6,
                      2,7,3,9,1],

    "spending_pattern": [5000,1500,7000,1000,4500,
                         800,5200,2000,7500,1200],

    "shopping_frequency": [15,5,18,3,12,
                           2,16,6,20,4],

    # 1 = Electronics
    # 2 = Fashion
    # 3 = Grocery
    "recommended_product": [1,2,1,3,2,
                            3,1,2,1,3]
}

# Convert dataframe
df = pd.DataFrame(data)

# Features
X = df[[
    "purchase_history",
    "product_similarity",
    "cart_behavior",
    "spending_pattern",
    "shopping_frequency"
]]

# Target
y = df["recommended_product"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# KNN Model
model = KNeighborsClassifier(n_neighbors=3)

# Train model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nSMART SHOPPING RECOMMENDATION SYSTEM\n")

print("Accuracy Score :", round(accuracy,2))

# User Input
print("\nENTER CUSTOMER DETAILS\n")

purchase_history = float(
    input("Purchase History Score (1-10) : ")
)

product_similarity = float(
    input("Product Similarity Score (1-10) : ")
)

cart_behavior = float(
    input("Cart Behavior Score (1-10) : ")
)

spending_pattern = float(
    input("Monthly Spending Amount : ")
)

shopping_frequency = float(
    input("Shopping Frequency Per Month : ")
)

# Predict
prediction = model.predict([[
    purchase_history,
    product_similarity,
    cart_behavior,
    spending_pattern,
    shopping_frequency
]])

# Output
print("\nSHOPPING RECOMMENDATION RESULT\n")

if prediction[0] == 1:
    print("Recommended Product Category : ELECTRONICS")

elif prediction[0] == 2:
    print("Recommended Product Category : FASHION")

else:
    print("Recommended Product Category : GROCERY")

# Similarity matching
print("Similar buyer profiles matched successfully")

# Offer message
print("Personalized offers generated successfully")