import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Dataset
data = {
    "genre": [1,2,1,3,2,
              1,3,2,1,3],

    "listening_hours": [2,5,3,6,4,
                        2,7,5,3,6],

    "favorite_artist_type": [1,2,1,3,2,
                             1,3,2,1,3],

    "mood_preference": [1,2,1,3,2,
                        1,3,2,1,3],

    # 1 = Melody Playlist
    # 2 = Pop Playlist
    # 3 = Rock Playlist
    "recommended_playlist": [1,2,1,3,2,
                             1,3,2,1,3]
}

# Convert dataframe
df = pd.DataFrame(data)

# Features
X = df[[
    "genre",
    "listening_hours",
    "favorite_artist_type",
    "mood_preference"
]]

# Target
y = df["recommended_playlist"]

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

print("\nMUSIC TASTE RECOMMENDATION SYSTEM\n")

print("Accuracy Score :", round(accuracy,2))

# User Input
print("\nENTER USER MUSIC DETAILS\n")

genre = int(input(
    "Genre Preference (1 Melody / 2 Pop / 3 Rock) : "
))

listening_hours = float(
    input("Daily Listening Hours : ")
)

favorite_artist_type = int(input(
    "Favorite Artist Type (1 Melody / 2 Pop / 3 Rock) : "
))

mood_preference = int(input(
    "Mood Preference (1 Calm / 2 Happy / 3 Energetic) : "
))

# Predict
prediction = model.predict([[
    genre,
    listening_hours,
    favorite_artist_type,
    mood_preference
]])

# Output
print("\nRECOMMENDATION RESULT\n")

if prediction[0] == 1:
    print("Recommended Playlist : MELODY SONGS")

elif prediction[0] == 2:
    print("Recommended Playlist : POP SONGS")

else:
    print("Recommended Playlist : ROCK SONGS")

# Similarity message
print("Similar listener profiles matched successfully")