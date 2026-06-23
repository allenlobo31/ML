import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_california_housing

california = fetch_california_housing()
df = pd.DataFrame(california.data, columns=california.feature_names)

df.hist(bins=30, figsize=(12, 10), edgecolor="black")
plt.suptitle("Histograms of Numerical Features")
plt.show()

plt.figure(figsize=(12, 8))
for i, column in enumerate(df.columns):
  plt.subplot(3, 3, i+1)
  sns.boxplot(y=df[column])
  plt.title(column)
plt.suptitle("\n\nBox Plots of Numerical Features")
plt.tight_layout()
plt.show()