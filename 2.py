import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

df = fetch_california_housing(as_frame=True).frame

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True,  fmt=".2f") #cmap='coolwarm
plt.title("Correlation Matrix Heatmap")
plt.show()

sns.pairplot(df.sample(500, random_state=42), diag_kind="kde")
plt.show()