import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Generate seismic data
np.random.seed(10)

dense_region = np.random.randn(400, 2) * 0.5 + [20, 70]
small_region = np.random.randn(100, 2) * 0.3 + [25, 75]
outliers = np.array([
    [40, 90],
    [5, 40]
])

data = np.vstack((dense_region, small_region, outliers))

df = pd.DataFrame(data, columns=['Latitude', 'Longitude'])

# DBSCAN
db = DBSCAN(eps=1.0, min_samples=10)
df['Cluster'] = db.fit_predict(df[['Latitude', 'Longitude']])

# Visualization
plt.figure(figsize=(8,6))
plt.scatter(df['Longitude'], df['Latitude'], c=df['Cluster'], cmap='viridis')
plt.title("Earthquake Dense Region Detection")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

# Risk Areas
risk = df[df['Cluster'] == -1]
print("\nPotential Risk Areas:")
print(risk)