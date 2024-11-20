import numpy as np
from sklearn.datasets import load_digits
from scipy.ndimage import shift
import matplotlib.pyplot as plt

def augment_mnist_with_shift(images, labels, num_augmented=10000):
    """
    Augments the MNIST dataset by randomly shifting images.
    
    Parameters:
        images (numpy.ndarray): The original images of shape (n_samples, height, width).
        labels (numpy.ndarray): The corresponding labels of the images.
        num_augmented (int): Number of new images to generate.
    
    Returns:
        numpy.ndarray: Augmented images.
        numpy.ndarray: Corresponding labels for the augmented images.
    """
    augmented_images = []
    augmented_labels = []

    for _ in range(num_augmented):
        # Select a random image
        random_index = np.random.randint(0, len(images))
        original_image = images[random_index]
        label = labels[random_index]

        # Generate random shift values (between -2 and +2 for x and y)
        shift_x = np.random.choice([-2, -1, 1, 2])
        shift_y = np.random.choice([-2, -1, 1, 2])
        
        # Apply the shift to the image
        shifted_image = shift(original_image, shift=(shift_x, shift_y), mode='constant', cval=0)

        # Append the shifted image and its label
        augmented_images.append(shifted_image)
        augmented_labels.append(label)

    # Convert to numpy arrays
    augmented_images = np.array(augmented_images)
    augmented_labels = np.array(augmented_labels)

    return augmented_images, augmented_labels

# Load the MNIST dataset (example with sklearn's load_digits for simplicity)
data = load_digits()
images = data.images  # Shape: (1797, 8, 8)
labels = data.target

# Extend the dataset
augmented_images, augmented_labels = augment_mnist_with_shift(images, labels, num_augmented=500)

# Display original and augmented images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(images[0], cmap='gray')
plt.title("Original Image")
plt.subplot(1, 2, 2)
plt.imshow(augmented_images[0], cmap='gray')
plt.title("Shifted Image")
plt.show()
