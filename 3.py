import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# 1. Load data and extract names
iris = load_iris()

# 2. Apply PCA to reduce features from 4 to 2
pca_data = PCA(n_components=2).fit_transform(iris.data)

# 3. Create a DataFrame for easy plotting
df = pd.DataFrame(pca_data, columns=["PC1", "PC2"])
df["Species"] = [iris.target_names[i] for i in iris.target]

# 4. Plot using Seaborn
sns.scatterplot(data=df, x="PC1", y="PC2", hue="Species", palette="Set1")
plt.title("PCA on Iris Dataset")
plt.grid(True, alpha=0.3)
plt.show()