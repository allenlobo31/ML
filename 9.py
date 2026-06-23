import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

# 1. Load the Olivetti Faces dataset
faces = fetch_olivetti_faces(shuffle=True, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    faces.data, faces.target, test_size=0.3, random_state=42
)

# 2. Train the Gaussian Naive Bayes Classifier
gnb = GaussianNB().fit(X_train, y_train)
print(f"Test Accuracy: {gnb.score(X_test, y_test) * 100:.2f}%")

# 3. Predict and Visualize the first 10 test samples
y_pred = gnb.predict(X_test)

print(confusion_matrix(y_test, y_pred))
fig, axes = plt.subplots(3, 5, figsize=(12, 6))
for ax, img, true_lbl, pred_lbl in zip(axes.ravel(), X_test, y_test, y_pred):
    ax.imshow(img.reshape(64, 64), cmap="gray")
    ax.set_title(f"T: {true_lbl} | P: {pred_lbl}")
    ax.axis("off")

plt.tight_layout()
plt.show()