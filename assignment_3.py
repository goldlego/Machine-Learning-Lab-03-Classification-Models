import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

# PART 1: SETUP
print("--- Data Prep ---")

df = pd.read_csv("Lab Session Data.xlsx - marketing_campaign.csv")
features = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
target = 'Response'

df = df.dropna()

X = df[features].values
y = df[target].values

print(f"Shape: {X.shape}")

# A6: 70/30 Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# PART 2: CUSTOM FUNCTIONS

# A1, A4, A5: Vector Math
def custom_dot_product(v1, v2):
    return np.sum(v1 * v2)

def custom_euclidean_norm(v):
    return np.sqrt(np.sum(v ** 2))

def custom_minkowski_distance(v1, v2, p):
    return np.power(np.sum(np.abs(v1 - v2) ** p), 1/p)

# A2: Statistics
def custom_mean(data):
    if len(data) == 0: return 0.0
    return np.sum(data, axis=0) / len(data)

def custom_variance(data):
    if len(data) == 0: return 0.0
    mu = custom_mean(data)
    return np.sum((data - mu) ** 2, axis=0) / len(data)

def custom_std_dev(data):
    return np.sqrt(custom_variance(data))

# A10: k-NN Logic
def custom_knn_predict(X_train, y_train, test_vector, k=3):
    # Vectorized Euclidean distance
    distances = np.sqrt(np.sum((X_train - test_vector) ** 2, axis=1))
    
    # Top k neighbors
    k_indices = np.argsort(distances)[:k]
    k_labels = y_train[k_indices]
    
    # Majority vote
    values, counts = np.unique(k_labels, return_counts=True)
    return values[np.argmax(counts)]

# A13: Metrics
def custom_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def custom_precision_recall_f1(y_true, y_pred, pos_label=1):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
    fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
    fn = np.sum((y_true == pos_label) & (y_pred != pos_label))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
    return precision, recall, f1

# A14: Matrix Inversion (Normal Equation)
def matrix_inversion_classifier(X_train, y_train, X_test):
    # Add bias term
    X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
    X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]
    
    # Weights = (X^T * X)^-1 * X^T * y
    weights = np.linalg.pinv(X_train_b.T @ X_train_b) @ X_train_b.T @ y_train
    
    return np.where(X_test_b @ weights >= 0.5, 1, 0)

# PART 3: EXECUTION

print("\n--- A1: Vectors ---")
v1, v2 = X[0], X[1]
print(f"Dot: {custom_dot_product(v1, v2)}")
print(f"Norm (v1): {custom_euclidean_norm(v1)}")
print(f"Np Dot: {np.dot(v1, v2)}")
print(f"Np Norm: {np.linalg.norm(v1)}")

print("\n--- A2: Stats ---")
c0, c1 = X[y == 0], X[y == 1]
mean0, mean1 = custom_mean(c0), custom_mean(c1)
print(f"Dist Centroids: {np.linalg.norm(mean0 - mean1):.4f}")
print(f"Intraclass Spread (C0): {np.mean(custom_std_dev(c0)):.4f}")

print("\n--- A3: Histogram ---")
plt.hist(df['MntWines'], bins=30, color='skyblue', edgecolor='black')
plt.title('A3: Wine Spending')
plt.show()

print("\n--- A4: Minkowski Plot ---")
plt.plot(range(1, 11), [custom_minkowski_distance(v1, v2, p) for p in range(1, 11)], marker='o')
plt.title('A4: Distance vs p')
plt.grid(True)
plt.show()

print("\n--- A5: Distance Check ---")
d_scipy = distance.minkowski(v1, v2, 3)
d_custom = custom_minkowski_distance(v1, v2, 3)
print(f"Scipy: {d_scipy}, Custom: {d_custom}")
if abs(d_scipy - d_custom) < 1e-5: print(">> Match")

print("\n--- A7/A8: Sklearn k-NN ---")
neigh = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
acc_sk = neigh.score(X_test, y_test)
print(f"Accuracy: {acc_sk:.4f}")

print("\n--- A10: Custom k-NN Inference ---")
preds = np.array([custom_knn_predict(X_train, y_train, x, k=3) for x in X_test])
acc_cust = custom_accuracy(y_test, preds)
print(f"Accuracy: {acc_cust:.4f}")
if acc_sk == acc_cust: print(">> Match")

print("\n--- A11: Accuracy vs k ---")
accs = [KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train).score(X_test, y_test) for k in range(1, 12)]
plt.plot(range(1, 12), accs, marker='x', color='red')
plt.title('A11: Accuracy vs k')
plt.grid(True)
plt.show()

print("\n--- A12/A13: Metrics ---")
print(classification_report(y_test, preds))
p, r, f1 = custom_precision_recall_f1(y_test, preds)
print(f"Custom (Class 1) -> P: {p:.2f}, R: {r:.2f}, F1: {f1:.2f}")

print("\n--- A14: Matrix Inversion ---")
acc_inv = custom_accuracy(y_test, matrix_inversion_classifier(X_train, y_train, X_test))
print(f"Inv Accuracy: {acc_inv:.4f} vs k-NN: {acc_cust:.4f}")
