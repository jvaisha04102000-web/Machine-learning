import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("cab_fare_dataset.csv")

plt.scatter(df["distance"], df["fare"])

plt.xlabel("Distance")
plt.ylabel("Fare")

plt.title("Distance vs Fare")

plt.show()