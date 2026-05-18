import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv("shopping_users.csv")


features = df[['TimeSpent', 'PurchaseAmount', 'CategoriesVisited']]

# -----------------------------
# Scaling
# -----------------------------

scaler = StandardScaler()

scaled_features = scaler.fit_transform(features)

# -----------------------------
# Hierarchical Clustering
# -----------------------------

linkage_matrix = linkage(scaled_features, method='ward')

# -----------------------------
# Dendrogram
# -----------------------------

plt.figure(figsize=(10, 5))

dendrogram(linkage_matrix)

plt.title("Online Shopping User Hierarchy")
plt.xlabel("Users")
plt.ylabel("Distance")

plt.show()