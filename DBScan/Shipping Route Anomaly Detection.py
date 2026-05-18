import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Simulated GPS route data
np.random.seed(20)

normal_route = np.random.randn(500, 2) * 0.5 + [30, 60]

suspicious_routes = np.array([
    [40, 80],
    [45, 85],
    [50, 90]
])

data = np.vstack((normal_route, suspicious_routes))

df = pd.DataFrame(data, columns=['Latitude', 'Longitude'])

# DBSCAN
model = DBSCAN(eps=1, min_samples=10)
df['Cluster'] = model.fit_predict(df[['Latitude', 'Longitude']])

# Plot
plt.figure(figsize=(8,6))
plt.scatter(df['Longitude'], df['Latitude'],
            c=df['Cluster'], cmap='Set1')

plt.title("Shipping Route Anomaly Detection")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

# Suspicious Routes
anomaly = df[df['Cluster'] == -1]
print("\nSuspicious Shipping Routes:")
print(anomaly)