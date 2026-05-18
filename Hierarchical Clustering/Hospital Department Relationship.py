import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv("hospital_data.csv")

# Example columns:
# Doctor, Specialization, Department, PatientCount

# -----------------------------
# Encode Categorical Data
# -----------------------------

encoder = LabelEncoder()

df['Specialization'] = encoder.fit_transform(df['Specialization'])
df['Department'] = encoder.fit_transform(df['Department'])

features = df[['Specialization', 'Department', 'PatientCount']]

# -----------------------------
# Scaling
# -----------------------------

scaler = StandardScaler()

scaled_features = scaler.fit_transform(features)

# -----------------------------
#  Hierarchical Clustering
# -----------------------------

linkage_matrix = linkage(scaled_features, method='ward')

# -----------------------------
#  Dendrogram
# -----------------------------

plt.figure(figsize=(12, 6))

dendrogram(linkage_matrix)

plt.title("Hospital Department Hierarchy")
plt.xlabel("Departments")
plt.ylabel("Distance")

plt.show()