from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Load the dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split the dataset: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline 1: Linear SVM with hinge loss (baseline)
pipeline1 = Pipeline([
    ('scaler', StandardScaler()),  # Standardize the features
    ('svm', LinearSVC(C=1, max_iter=1000, loss='hinge', random_state=42))  # Linear SVM
])

# Pipeline 2: Linear SVM with polynomial features (degree=3)
pipeline2 = Pipeline([
    ('scaler', StandardScaler()),  # Standardize the features
    ('poly', PolynomialFeatures(degree=3)),  # Generate polynomial features
    ('svm', LinearSVC(C=1, max_iter=1000, loss='hinge', random_state=42))  # Linear SVM
])

# Pipeline 3: SVM with a polynomial kernel (degree=3)
pipeline3 = Pipeline([
    ('scaler', StandardScaler()),  # Standardize the features
    ('svm', SVC(kernel='poly', degree=3, C=1, coef0=1, random_state=42))  # Polynomial kernel SVM
])

# Train and evaluate each pipeline
pipelines = {'Pipeline 1 (Linear SVM)': pipeline1,
             'Pipeline 2 (Poly Features + Linear SVM)': pipeline2,
             'Pipeline 3 (Poly Kernel SVM)': pipeline3}

for name, pipeline in pipelines.items():
    print(f"\n{name}:")
    
    # Train the pipeline
    pipeline.fit(X_train, y_train)
    
    # Test the pipeline
    y_pred = pipeline.predict(X_test)
    
    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))

###
#c
###
# Store accuracy results for comparison
accuracy_results = {}

# Evaluate each pipeline
for name, pipeline in pipelines.items():
    # Predict on the test set
    y_pred = pipeline.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_results[name] = accuracy

    print(f"\n{name}:")
    print(f"Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))

# Summary of accuracies
print("\n### Accuracy Comparison ###")
for name, accuracy in accuracy_results.items():
    print(f"{name}: {accuracy:.2f}")




###
#d
###
# Define the parameter grid
param_grid = {
    'C': np.logspace(-1, 2, 10),  # Range of C: 0.1 to 100 (logarithmic scale)
    'max_iter': [100, 500, 1000, 5000, 10000]  # Iterations range
}

# Instantiate the classifier (best from Pipeline 3)
svc = SVC(kernel='poly', degree=3, coef0=1, random_state=42)

# Set up the grid search
grid_search = GridSearchCV(
    estimator=svc,
    param_grid={
        'C': param_grid['C'], 
        'max_iter': param_grid['max_iter']
    },
    scoring='accuracy',
    cv=5,  # 5-fold cross-validation
    verbose=2,
    n_jobs=-1  # Use all available cores
)

# Perform the grid search
grid_search.fit(X_train, y_train)

# Print the best parameters and the corresponding accuracy
print("Best Parameters:")
print(grid_search.best_params_)
print(f"Best Cross-Validated Accuracy: {grid_search.best_score_:.2f}")

# Evaluate the best model on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

print(f"\nTest Accuracy of the Best Model: {test_accuracy:.2f}")
print(classification_report(y_test, y_pred))
































