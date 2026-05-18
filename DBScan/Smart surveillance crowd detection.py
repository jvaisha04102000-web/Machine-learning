import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Simulated CCTV coordinates
np.random.seed(5)

crowd1 = np.random.randn(200, 2) * 2 + [50, 50]
crowd2 = np.random.randn(150, 2) * 1.5 + [70, 70]
isolated_people = np.array([
    [20, 10],
    [90, 95]
])

data = np.vstack((crowd1, crowd2, isolated_people))

df = pd.DataFrame(data, columns=['X', 'Y'])

# DBSCAN
model = DBSCAN(eps=3, min_samples=8)
df['Cluster'] = model.fit_predict(df[['X', 'Y']])

# Plot
plt.figure(figsize=(8,6))
plt.scatter(df['X'], df['Y'], c=df['Cluster'], cmap='cool')
plt.title("Crowd Density Detection")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.show()

# Emergency Alerts
alerts = df[df['Cluster'] == -1]
print("\nEmergency Alert Points:")
print(alerts)