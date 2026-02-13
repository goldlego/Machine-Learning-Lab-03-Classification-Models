import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV

# --- FUNCTION DEFINITIONS (Requirement A1, A2, A3, A4, A7) ---

def evaluate_classification_metrics(y_true, y_pred):
    """A1: Evaluates confusion matrix and performance metrics."""
    cm = confusion_matrix(y_true, y_pred)
    # Using classification_report for precision, recall, and F1-score
    report = classification_report(y_true, y_pred, output_dict=True)
    return cm, report

def calculate_regression_metrics(y_true, y_pred):
    """A2: Calculates MSE, RMSE, MAPE, and R2."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    # MAPE calculation
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mape, r2

def generate_synthetic_data(n_points=20):
    """A3: Generates 20 random data points for two features."""
    X = np.random.uniform(1, 10, (n_points, 2))
    # Assign classes: class 0 if X+Y < 11, else class 1 
    y = np.where(X[:, 0] + X[:, 1] < 11, 0, 1)
    return X, y

def generate_test_grid():
    """A4: Generates a dense grid of test points between 0 and 10."""
    x_range = np.arange(0, 10.1, 0.1)
    y_range = np.arange(0, 10.1, 0.1)
    xx, yy = np.meshgrid(x_range, y_range)
    X_test = np.c_[xx.ravel(), yy.ravel()]
    return X_test, xx, yy

def find_best_k(X, y):
    """A7: Uses GridSearchCV to find the ideal k value."""
    parameters = {'n_neighbors': range(1, 21)}
    knn = KNeighborsClassifier()
    clf = GridSearchCV(knn, parameters, cv=5)
    clf.fit(X, y)
    return clf.best_params_['n_neighbors']

# --- MAIN EXECUTION BLOCK ---

# Load project data (Marketing Campaign)
df = pd.read_csv("Lab Session Data.xlsx - marketing_campaign.csv").dropna()
features = ['MntWines', 'MntMeatProducts'] # Using 2 features for visualization 
X_proj = df[features].values
y_proj = df['Response'].values
X_train, X_test, y_train, y_test = train_test_split(X_proj, y_proj, test_size=0.3, random_state=42)

# A1: Classification Metrics for Project Data
knn_proj = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
train_preds = knn_proj.predict(X_train)
test_preds = knn_proj.predict(X_test)

cm_train, report_train = evaluate_classification_metrics(y_train, train_preds)
cm_test, report_test = evaluate_classification_metrics(y_test, test_preds)

print("--- A1: Performance Metrics (Test Set) ---")
print(f"Confusion Matrix:\n{cm_test}")
print(f"Accuracy: {report_test['accuracy']:.2f}")
print(f"F1-Score (Class 1): {report_test['1']['f1-score']:.2f}")

# A3: Synthetic Training Data Plot
X_syn, y_syn = generate_synthetic_data()
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(X_syn[y_syn==0][:,0], X_syn[y_syn==0][:,1], c='blue', label='Class 0')
plt.scatter(X_syn[y_syn==1][:,0], X_syn[y_syn==1][:,1], c='red', label='Class 1')
plt.title("A3: Synthetic Training Data")
plt.legend()

# A4: KNN Prediction on Dense Grid
X_grid, xx, yy = generate_test_grid()
knn_syn = KNeighborsClassifier(n_neighbors=3).fit(X_syn, y_syn)
grid_preds = knn_syn.predict(X_grid)

# A4/A5: Visualize Class Boundaries
plt.subplot(1, 2, 2)
colors = np.where(grid_preds == 0, 'blue', 'red')
plt.scatter(X_grid[:, 0], X_grid[:, 1], c=colors, s=1, alpha=0.1)
plt.title("A4: KNN (k=3) Class Boundaries")
plt.show()

# A7: Hyper-parameter Tuning
best_k = find_best_k(X_train, y_train)
print(f"\n--- A7: Hyper-parameter Tuning ---")
print(f"Ideal 'k' value found: {best_k}")

