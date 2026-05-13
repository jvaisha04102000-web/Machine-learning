# Banking Customer Segmentation

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Dataset
data = {
    'Transactions': [10, 50, 15, 80, 25, 90, 30, 100],
    'CreditUsage': [20, 90, 25, 95, 35, 85, 40, 98],
    'Savings': [5000, 50000, 7000, 60000, 10000, 75000, 12000, 90000]
}

df = pd.DataFrame(data)

# Features
X = df[['Transactions', 'CreditUsage', 'Savings']]

# K-Means Model
model = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = model.fit_predict(X)

# Output
print(df)

# VIP Customer Identification
print("\nVIP Customers:")
vip = df[df['Savings'] > 50000]
print(vip)

# Visualization
plt.scatter(df['Transactions'], df['Savings'], c=df['Cluster'])
plt.xlabel("Transactions")
plt.ylabel("Savings")
plt.title("Customer Segmentation")
plt.show()