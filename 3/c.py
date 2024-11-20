import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = load_digits()
images = data.images
labels = data.target

# Flatten images for classifier
n_samples = len(images)
images_flat = images.reshape(n_samples, -1)

# Split data: 85% training, 15% testing
X_train, X_test, y_train, y_test = train_test_split(images_flat, labels, test_size=0.15, random_state=42)

# Augmentation function from part b
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

# Augmented training data
augmented_images, augmented_labels = augment_mnist_with_shift(X_train, y_train, num_augmented=1000)
X_train_extended = np.vstack([X_train, augmented_images])
y_train_extended = np.hstack([y_train, augmented_labels])

# Train and evaluate with original data
clf_original = DecisionTreeClassifier(random_state=42)
clf_original.fit(X_train, y_train)
y_pred_original = clf_original.predict(X_test)
accuracy_original = accuracy_score(y_test, y_pred_original)

print("\nModel with Original Training Data:")
print(f"Accuracy: {accuracy_original:.2f}")
print(classification_report(y_test, y_pred_original))

# Train and evaluate with augmented data
clf_augmented = DecisionTreeClassifier(random_state=42)
clf_augmented.fit(X_train_extended, y_train_extended)
y_pred_augmented = clf_augmented.predict(X_test)
accuracy_augmented = accuracy_score(y_test, y_pred_augmented)

print("\nModel with Augmented Training Data:")
print(f"Accuracy: {accuracy_augmented:.2f}")
print(classification_report(y_test, y_pred_augmented))
