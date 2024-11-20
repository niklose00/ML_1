import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import *
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the California housing dataset
data = fetch_california_housing(as_frame=True)
df = data.frame  # DataFrame containing features and target


###
#a
###

# Explore the data
print("Dataset Overview:")
print(df.head())

print("\nFeature Information:")
print(data.feature_names)

print("\nTarget Variable (Median House Value):")
print(df['MedHouseVal'].describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Compute correlations between attributes
print("\n### Correlation Matrix ###")
correlation_matrix = df.corr()
print(correlation_matrix)

# Visualize the correlations
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
plt.title("Correlation Matrix of California Housing Dataset")
plt.show()


#######
#b
#######
# Add the categorical attribute 'HOL'
df['HOL'] = ((df['HouseAge'] > 25) & (df['AveBedrms'] > 3)).astype(int)

# Verify the new column
print("Dataset with HOL attribute:")
print(df[['HouseAge', 'AveBedrms', 'HOL']].head())

# Count rows where HOL is 1
hol_count = df['HOL'].sum()
print(f"\nNumber of rows where HOL is 1: {hol_count}")

print(df)

#######
#c
#######
# Split the dataset into features (X) and target (y)
X = df.drop(columns=['MedHouseVal'])  # Features
y = df['MedHouseVal']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)



#######
#d
#######
# Initialize the Linear Regressor
linear_regressor = LinearRegression()

# Train the model on the training set
linear_regressor.fit(X_train, y_train)


#######
#e
#######
# Predictions
y_train_pred = linear_regressor.predict(X_train)
y_test_pred = linear_regressor.predict(X_test)

# Training set metrics
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)

# Test set metrics
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_test_pred)

# Print results
print("### Training Set Performance ###")
print(f"MSE: {train_mse:.2f}")
print(f"RMSE: {train_rmse:.2f}")
print(f"MAE: {train_mae:.2f}")

print("\n### Test Set Performance ###")
print(f"MSE: {test_mse:.2f}")
print(f"RMSE: {test_rmse:.2f}")
print(f"MAE: {test_mae:.2f}")





















