# Smart Tourism Recommendation Clustering

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Dataset
data = {
    'TravelHistory': [2, 10, 3, 15, 5, 12, 4, 18],
    'Budget': [20000, 100000, 30000, 150000, 40000, 120000, 35000, 170000],
    'DestinationType': [1, 3, 1, 4, 2, 3, 2, 4]
}

df = pd.DataFrame(data)

# Features
X = df[['TravelHistory', 'Budget', 'DestinationType']]

# K-Means
model = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = model.fit_predict(X)

# Output
print(df)

# Personalized Package Suggestion
for cluster in df['Cluster']:
    if cluster == 0:
        print("Budget Travel Package")
    elif cluster == 1:
        print("Luxury International Package")
    else:
        print("Family Vacation Package")

# Visualization
plt.scatter(df['TravelHistory'], df['Budget'], c=df['Cluster'])
plt.xlabel("Travel History")
plt.ylabel("Budget")
plt.title("Tourist Clustering")
plt.show()