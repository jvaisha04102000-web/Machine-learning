import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# User activity data
np.random.seed(7)

real_users = np.random.randn(300, 2) * 5 + [50, 100]
fake_users = np.array([
    [150, 300],
    [170, 280],
    [200, 350]
])

data = np.vstack((real_users, fake_users))

df = pd.DataFrame(data, columns=['Posts_Per_Day', 'Friend_Requests'])

# DBSCAN
db = DBSCAN(eps=10, min_samples=6)
df['Cluster'] = db.fit_predict(df[['Posts_Per_Day', 'Friend_Requests']])

# Visualization
plt.figure(figsize=(8,6))
plt.scatter(df['Posts_Per_Day'], df['Friend_Requests'],
            c=df['Cluster'], cmap='plasma')

plt.title("Fake Account Detection")
plt.xlabel("Posts Per Day")
plt.ylabel("Friend Requests Sent")
plt.show()

# Fake Accounts
fake = df[df['Cluster'] == -1]
print("\nDetected Fake Accounts:")
print(fake)