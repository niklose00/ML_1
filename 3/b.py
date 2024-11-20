import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the MNIST-like dataset (using sklearn's digits for simplicity)
data = load_digits()
images = data.images
labels = data.target

# Flatten the images for the classifier
n_samples = len(images)
images_flat = images.reshape(n_samples, -1)  # Shape: (n_samples, n_features)

# Split the data: 85% training, 15% test
X_train, X_test, y_train, y_test = train_test_split(images_flat, labels, test_size=0.15, random_state=42)

# Augment the training data with the augmentation function
from scipy.ndimage import shift

def augment_mnist_with_shift(images, labels, num_augmented=1000):
    augmented_images = []
    augmented_labels = []
    for _ in range(num_augmented):
        random_index = np.random.randint(0, len(images))
        original_image = images[random_index].reshape(8, 8)
        label = labels[random_index]
        shift_x = np.random.choice([-2, -1, 1, 2])
        shift_y = np.random.choice([-2, -1, 1, 2])
        shifted_image = shift(original_image, shift=(shift_x, shift_y), mode='constant', cval=0).flatten()
        augmented_images.append(shifted_image)
        augmented_labels.append(label)
    return np.array(augmented_images), np.array(augmented_labels)

# Generate augmented data
augmented_images, augmented_labels = augment_mnist_with_shift(X_train, y_train, num_augmented=1000)

# Combine original and augmented training data
X_train_extended = np.vstack([X_train, augmented_images])
y_train_extended = np.hstack([y_train, augmented_labels])

# Train a decision tree classifier on the extended training data
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train_extended, y_train_extended)

# Evaluate the classifier on the test set
y_pred = clf.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the Decision Tree Classifier: {accuracy:.2f}")

# Detailed metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
