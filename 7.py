import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

# --- 1. Linear Regression (California Housing) ---
cali = fetch_california_housing(as_frame=True)
X, y = cali.data[["AveRooms"]], cali.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression().fit(X_tr, y_tr)

plt.scatter(X_te, y_te, color="blue", alpha=0.5, label="Actual")
plt.plot(X_te, lr.predict(X_te), color="red", label="Linear Fit")
plt.title("Linear Regression (Rooms vs Price)")
plt.legend()
plt.show()

# --- 2. Polynomial Regression (Auto MPG) ---
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
df = pd.read_csv(
    url,
    sep=r"\s+",
    names=["mpg", "cyl", "disp", "hp", "wt", "acc", "year", "origin"],
    na_values="?",
).dropna()
X_p, y_p = df[["disp"]], df["mpg"]
X_ptr, X_pte, y_ptr, y_pte = train_test_split(
    X_p, y_p, test_size=0.2, random_state=42
)

# Pipeline to automatically create X^2 features and fit the linear model
poly = make_pipeline(PolynomialFeatures(degree=2), LinearRegression()).fit(
    X_ptr, y_ptr
)

# Sort test inputs to draw a perfectly smooth continuous curve line
sort_idx = np.argsort(X_pte.values.flatten())
X_pte_sorted = X_pte.values[sort_idx]

plt.scatter(X_pte, y_pte, color="blue", alpha=0.5, label="Actual")
plt.plot(X_pte_sorted, poly.predict(X_pte_sorted), color="red", label="Poly Fit")
plt.title("Polynomial Regression (Engine Displacement vs MPG)")
plt.legend()
plt.show()