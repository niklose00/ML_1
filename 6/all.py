from sklearn.datasets import fetch_openml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_text, plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#####
#a
#####
# Fetch the Cholesterol dataset from OpenML
cholesterol_data = fetch_openml(data_id=204, as_frame=True)

# Extract data and target
df = cholesterol_data.frame

# Explore the dataset
print("### Dataset Overview ###")
print(df.head())


# Check for missing values
print("\n### Missing Values ###")
print(df.isnull().sum())

# Feature and target columns
print("\n### Features and Target ###")
print(f"Feature Names: {cholesterol_data.feature_names}")
print(f"Target Name: {cholesterol_data.target.name}")

#####
#b
#####
# Define features (X) and target (y)
X = df[cholesterol_data.feature_names]
y = df[cholesterol_data.target.name]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Decision Tree Regressor with max_depth=2
tree_regressor = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_regressor.fit(X_train, y_train)

# Visualize the Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(tree_regressor, feature_names=X.columns, filled=True, rounded=True)
plt.title("Decision Tree Regressor (max_depth=2)")
plt.show()

# Text-based visualization
tree_rules = export_text(tree_regressor, feature_names=list(X.columns))
print("\nDecision Tree Rules:")
print(tree_rules)






#####
#c
#####
# Define the parameter grid
param_grid = {
    'max_depth': range(2, 16),               # Depth values from 2 to 15
    'criterion': ['squared_error', 'absolute_error'],  # Split quality criteria
    'splitter': ['best', 'random']          # Splitter: Best or Random
}

# Initialize the Decision Tree Regressor
tree_regressor = DecisionTreeRegressor(random_state=42)

# Set up the GridSearchCV
grid_search = GridSearchCV(
    estimator=tree_regressor,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',  # Minimize MSE
    cv=5,                              # 5-fold cross-validation
    verbose=2,
    n_jobs=-1                          # Use all available CPU cores
)

# Perform the grid search on the training data
grid_search.fit(X_train, y_train)

# Output the best parameters and corresponding score
print("Best Parameters:")
print(grid_search.best_params_)
print(f"Best Cross-Validated MSE: {-grid_search.best_score_:.2f}")

# Evaluate the best model on the test set
best_model = grid_search.best_estimator_
y_test_pred = best_model.predict(X_test)

# Test set metrics
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = test_mse ** 0.5
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\nTest Set Performance:")
print(f"MSE: {test_mse:.2f}")
print(f"RMSE: {test_rmse:.2f}")
print(f"MAE: {test_mae:.2f}")
print(f"R² Score: {test_r2:.2f}")


#####
#d
#####






























