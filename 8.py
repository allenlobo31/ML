import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# 1. Load data and split into train/test
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=42
)

# 2. Train Decision Tree (limiting depth so the plot is readable)
clf = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_train, y_train)
print(f"Model Accuracy: {clf.score(X_test, y_test) * 100:.2f}%")

# 3. Classify a new sample
new_sample = [X_test[0]]
pred_name = cancer.target_names[clf.predict(new_sample)[0]]
print(f"Predicted Class for the new sample: {pred_name}")

# 4. Plot the tree structure
plt.figure(figsize=(15, 8))
tree.plot_tree(
    clf, filled=True, feature_names=cancer.feature_names, class_names=cancer.target_names
)
plt.show()