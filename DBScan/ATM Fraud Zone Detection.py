import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Sample ATM transaction location data
np.random.seed(42)

normal_transactions = np.random.randn(300, 2) * 0.01 + [13.0827, 80.2707]
fraud_transactions = np.array([
    [13.20, 80.45],
    [12.90, 80.10],
    [13.30, 80.60]
])

data = np.vstack((normal_transactions, fraud_transactions))

df = pd.DataFrame(data, columns=['Latitude', 'Longitude'])

# DBSCAN Model
model = DBSCAN(eps=0.02, min_samples=5)
df['Cluster'] = model.fit_predict(df[['Latitude', 'Longitude']])

# Plot
plt.figure(figsize=(8,6))
plt.scatter(df['Longitude'], df['Latitude'], c=df['Cluster'], cmap='rainbow')
plt.title("ATM Fraud Zone Detection")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

# Outliers
frauds = df[df['Cluster'] == -1]
print("\nFraud/Outlier Transactions:")
print(frauds)