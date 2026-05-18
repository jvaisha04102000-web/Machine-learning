import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv("employee_skills.csv")


features = df[['Python', 'SQL', 'Excel',
               'MachineLearning', 'Communication']]

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

plt.figure(figsize=(12, 6))

dendrogram(linkage_matrix)

plt.title("Employee Skill Hierarchy")
plt.xlabel("Employees")
plt.ylabel("Distance")

plt.show()