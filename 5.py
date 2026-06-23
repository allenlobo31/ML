import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# 1. Generate data (100 random points)
np.random.seed(42)
x = np.random.rand(100, 1)  
train_x, test_x = x[:50], x[50:]
train_y = np.where(train_x.flatten() <= 0.5, "Class1", "Class2")

# 2. Train and Plot train
k_values = [1, 2, 3, 4, 5, 20, 30]
plt.figure(figsize=(10, 6))
colors = np.where(train_y == "Class1", "blue", "red")
plt.scatter(train_x, np.zeros(50), c=colors, marker="o", label="Train Data")

#3. Predict and plot
for i, k in enumerate(k_values, start=1):
    knn = KNeighborsClassifier(n_neighbors=k).fit(train_x, train_y)
    preds = knn.predict(test_x)

    # Plot predictions 
    test_colors = np.where(preds == "Class1", "blue", "red")
    plt.scatter(test_x, np.full(50, i), c=test_colors, marker="x")
    print(f"k={k} predictions: {list(preds[:])}")

#4. final graph
plt.yticks(range(len(k_values) + 1), ["Train"] + [f"k={k}" for k in k_values])
plt.xlabel("x value")
plt.title("KNN Classification for different k values")
plt.show()