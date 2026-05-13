# Hospital Patient Group Analysis

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Dataset
data = {
    'Age': [25, 70, 30, 80, 40, 75, 35, 85],
    'DiseaseHistory': [1, 5, 2, 6, 3, 5, 2, 7],
    'TreatmentCost': [10000, 90000, 15000, 120000, 30000, 100000, 25000, 150000]
}

df = pd.DataFrame(data)

# Features
X = df[['Age', 'DiseaseHistory', 'TreatmentCost']]

# K-Means Model
model = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = model.fit_predict(X)

# Output
print(df)

# High-Risk Group Identification
print("\nHigh Risk Patients:")
high_risk = df[df['TreatmentCost'] > 80000]
print(high_risk)

# Visualization
plt.scatter(df['Age'], df['TreatmentCost'], c=df['Cluster'])
plt.xlabel("Age")
plt.ylabel("Treatment Cost")
plt.title("Patient Group Clustering")
plt.show()