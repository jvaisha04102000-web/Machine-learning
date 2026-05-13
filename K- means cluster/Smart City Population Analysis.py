# Smart City Population Analysis

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Dataset
data = {
    'PopulationDensity': [1000, 9000, 2000, 12000, 3000, 10000, 2500, 15000],
    'TrafficMovement': [100, 900, 150, 1100, 200, 950, 180, 1300],
    'ResidentialPattern': [1, 3, 1, 4, 2, 3, 2, 5]
}

df = pd.DataFrame(data)

# Features
X = df[['PopulationDensity', 'TrafficMovement', 'ResidentialPattern']]

# K-Means Model
model = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = model.fit_predict(X)

# Output
print(df)

# Area Classification
for cluster in df['Cluster']:
    if cluster == 0:
        print("Residential Area")
    elif cluster == 1:
        print("Commercial Area")
    else:
        print("Industrial Area")

# Visualization
plt.scatter(df['PopulationDensity'], df['TrafficMovement'], c=df['Cluster'])
plt.xlabel("Population Density")
plt.ylabel("Traffic Movement")
plt.title("City Zone Clustering")
plt.show()