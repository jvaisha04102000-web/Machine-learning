import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Dataset
data = {
    "budget": [50,80,40,100,60,120,70,150,90,110],
    
    "actor_popularity": [7,9,6,10,8,9,7,10,8,9],
    
    "genre_score": [6,8,5,9,7,8,6,10,7,9],
    
    "trailer_views": [10,25,8,40,15,50,20,70,30,45],
    
    "social_media": [15,35,10,50,20,60,25,80,40,55],
    
    "revenue": [120,220,90,350,160,420,200,600,280,450]
}

# Convert to dataframe
df = pd.DataFrame(data)

# Features
X = df[[
    "budget",
    "actor_popularity",
    "genre_score",
    "trailer_views",
    "social_media"
]]

# Target
y = df["revenue"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Accuracy metrics
mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

r2 = r2_score(y_test, y_pred)

print("\nMOVIE REVENUE PREDICTION SYSTEM\n")

print("MAE :", round(mae,2))
print("RMSE :", round(rmse,2))
print("R2 Score :", round(r2,2))

# User Input
print("\nENTER MOVIE DETAILS\n")

budget = float(input("Movie Budget (Crores): "))

actor_popularity = float(input("Actor Popularity Score (1-10): "))

genre_score = float(input("Genre Score (1-10): "))

trailer_views = float(input("Trailer Views (Millions): "))

social_media = float(input("Social Media Engagement: "))

# Prediction
prediction = model.predict([[
    budget,
    actor_popularity,
    genre_score,
    trailer_views,
    social_media
]])

predicted_revenue = round(prediction[0],2)

print("\nPredicted Movie Revenue :", predicted_revenue, "Crores")

# Revenue analysis
if predicted_revenue > 400:
    print("Blockbuster Prediction")
    
elif predicted_revenue > 200:
    print("Hit Movie Prediction")
    
else:
    print("Average Revenue Prediction")