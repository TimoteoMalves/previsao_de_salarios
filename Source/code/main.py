import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import uniform, randint
from math import exp, log


dataset_path = "C:\\Users\\timot\\OneDrive\\Área de Trabalho\\Aulas 2025\\inteligencia artificial 2\\dataset.csv"

# --- 1. Load Data and Select Features ---
# Assume your dataset file is named 'ds_salaries.csv'
try:
    df = pd.read_csv(dataset_path)
except FileNotFoundError:
    print("Error: ds_salaries.csv not found. Please check the file path.")
    exit()

# Features for prediction (X) and Target variable (y)
FEATURES = [
    'experience_level', 
    'job_title', 
    'company_size', 
    'work_year',
    'employee_residence', 
    'company_location', 
    'remote_ratio' 
]
TARGET = 'salary_in_usd'

df_model = df[FEATURES + [TARGET]].copy()

# A. Convert 'remote_ratio' from percentage (0, 50, 100) to categorical string
df_model['remote_ratio'] = df_model['remote_ratio'].astype(str)

# B. Log Transform the Target Variable (Salary)
# This reduces the impact of outliers and normalizes the target distribution.
# Add 1 before log to handle any potential zero values, though salaries are > 0.
df_model[TARGET] = np.log1p(df_model[TARGET]) 

# C. One-Hot Encoding for all Categorical Features
# 'drop_first=True' helps prevent multicollinearity.
categorical_features = [
    'experience_level', 
    'job_title', 
    'company_size', 
    'employee_residence', 
    'company_location', 
    'remote_ratio'
]
df_encoded = pd.get_dummies(df_model, columns=categorical_features, drop_first=True)

# Remove the original target column and define X and y
X = df_encoded.drop(columns=[TARGET])
y = df_encoded[TARGET]

# --- 2. Splitting Data ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {len(X_train)} | Testing set size: {len(X_test)}")
print(f"Total features after encoding: {X.shape[1]}")

# --- 3. Hyperparameter Tuning (Randomized Search) ---

# Define the model and the parameter distribution for the search
rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)

param_dist = {
    'n_estimators': randint(100, 200, 300),    # Number of trees
    'max_depth': randint(10, 50),              # Max depth of each tree
    'min_samples_split': randint(2, 10),       # Min samples required to split a node
    'min_samples_leaf': randint(1, 5),         # Min samples required at a leaf node
    'max_features': ['sqrt', 'log2', 0.8]      # Number of features to consider for best split
}

# Use RandomizedSearchCV to efficiently find good hyperparameters
random_search = RandomizedSearchCV(
    estimator=rf_base, 
    param_distributions=param_dist, 
    n_iter=5, 
    cv=3,
    verbose=0, 
    random_state=42, 
    n_jobs=-1
)

print("\nStarting Hyperparameter Tuning...")
random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_

print("Tuning complete. Best parameters found:")
print(random_search.best_params_)

# --- 4. Prediction and Evaluation ---

# Make predictions using the best model
y_pred_log = best_model.predict(X_test)

# Reverse the Log Transformation for meaningful MAE/R2 calculation
y_test_actual = np.expm1(y_test)
y_pred_actual = np.expm1(y_pred_log)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test_actual, y_pred_actual)
r2 = r2_score(y_test_actual, y_pred_actual)

print("\n--- Model Evaluation (After Tuning and Log-Transform) ---")
print(f"Mean Absolute Error (MAE): ${mae:,.2f} (Average prediction error in USD)")
print(f"R-squared (R2) Score: {r2:.4f} (Variance explained)")

# --- 5. Visualization ---

# A. Feature Importance Plot (using the best model)
importances = best_model.feature_importances_
feature_names = X.columns
# Combine feature importances with their names and select top 15
forest_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(15)

plt.figure(figsize=(10, 7))
sns.barplot(x=forest_importances, y=forest_importances.index)
plt.title('Top 15 Random Forest Feature Importance')
plt.xlabel('Relative Importance')
plt.tight_layout()
plt.show()

# B. Actual vs. Predicted Salaries Plot
plt.figure(figsize=(8, 8))
plt.scatter(y_test_actual, y_pred_actual, alpha=0.6)
# Define a range for the diagonal line based on actual values
min_val = min(y_test_actual.min(), y_pred_actual.min())
max_val = max(y_test_actual.max(), y_pred_actual.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2) 
plt.xlabel('Actual Salaries (USD)')
plt.ylabel('Predicted Salaries (USD)')
plt.title('Actual vs. Predicted Salaries (USD)')
plt.grid(True)
plt.show()
