import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Load and Scale Data
cancer = load_breast_cancer()
X_scaled = StandardScaler().fit_transform(cancer.data)

# 2. Run K-Means Clustering (2 clusters: Malignant vs Benign)
kmeans = KMeans(n_clusters=2, n_init="auto", random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# 3. Reduce dimensions to 2D for plotting
pca_data = PCA(n_components=2).fit_transform(X_scaled)
df = pd.DataFrame(pca_data, columns=["PC1", "PC2"])
df["Cluster"] = clusters
df["True_Label"] = cancer.target

# Transform centroids into the same 2D PCA space
centroids = PCA(n_components=2).fit(X_scaled).transform(kmeans.cluster_centers_)

# 4. Visualize Side-by-Side Comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot K-Means Results + Centroids
sns.scatterplot(data=df, x="PC1", y="PC2", hue="Cluster", palette="Set1", ax=ax1)
ax1.scatter(centroids[:, 0], centroids[:, 1], c="black", marker="X", s=200, label="Centroids")
ax1.set_title("K-Means Clusters & Centroids")

# Plot Actual Medical Labels
sns.scatterplot(data=df, x="PC1", y="PC2", hue="True_Label", palette="coolwarm", ax=ax2)
ax2.set_title("Actual Diagnosis (True Labels)")

plt.show()