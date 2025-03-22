# Task A
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset from a folder
def load_images(folder):
    images, labels = [], []  # Lists to store images and corresponding labels
    for category in ["with_mask", "without_mask"]:  # Two categories: with and without a mask
        path = os.path.join(folder, category)  # Construct the path to the category folder
        label = 1 if category == "with_mask" else 0  # Assign labels: 1 for "with_mask", 0 for "without_mask"
        for file in os.listdir(path):  # Iterate through each file in the category folder
            img = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
            img = cv2.resize(img, (64, 64))  # Resize image to a standard size (64x64)
            images.append(img)  # Append the processed image to the list
            labels.append(label)  # Append the corresponding label
    return np.array(images), np.array(labels)  # Return images and labels as NumPy arrays

# Extract Features using HOG, Sobel, and Canny edge detection
def extract_features(images):
    features = []  # List to store feature vectors
    for img in images:
        # HOG (Histogram of Oriented Gradients) Feature Extraction
        hog_feature = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)

        # Sobel Edge Detection (Gradient in x and y directions)
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # Sobel filter in x-direction
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # Sobel filter in y-direction
        sobel_combined = cv2.magnitude(sobelx, sobely).flatten()  # Compute gradient magnitude and flatten

        # Canny Edge Detection
        canny_edges = cv2.Canny(img, 100, 200).flatten()  # Apply Canny edge detection and flatten

        # Concatenate all features into a single vector
        feature_vector = np.hstack([hog_feature, sobel_combined, canny_edges])
        features.append(feature_vector)  # Append feature vector to list
    
    return np.array(features)  # Return all feature vectors as a NumPy array

# Load dataset
dataset_path = "Face-Mask-Detection-master/dataset" # Path to the dataset folder
X, y = load_images(dataset_path)  # Load images and corresponding labels

# Extract Features from images
X_features = extract_features(X)

# Split dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)

# Standardize features to have zero mean and unit variance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit and transform training data
X_test = scaler.transform(X_test)  # Transform test data using the same scaler

# Train Support Vector Machine (SVM) Classifier
svm = SVC(kernel="linear")  # Use a linear kernel for SVM
svm.fit(X_train, y_train)  # Train the SVM model
y_pred_svm = svm.predict(X_test)  # Make predictions on the test set

# Train Neural Network (MLP - Multi-Layer Perceptron)
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)  # Define an MLP with 1 hidden layer of 100 neurons
mlp.fit(X_train, y_train)  # Train the MLP model
y_pred_mlp = mlp.predict(X_test)  # Make predictions on the test set

# Evaluate models using accuracy and classification reports
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))  # Print accuracy of SVM
print("MLP Accuracy:", accuracy_score(y_test, y_pred_mlp))  # Print accuracy of MLP

# Print detailed classification reports
print("\nSVM Classification Report:\n", classification_report(y_test, y_pred_svm))
print("\nMLP Classification Report:\n", classification_report(y_test, y_pred_mlp))


# Task B
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import os

# Define dataset paths
data_dir = "Face-Mask-Detection-master/dataset"

# Image parameters
img_size = (128, 128)  # Image size for input
batch_size = 32  # Balanced batch size for GPU and CPU training

# Data augmentation - helps improve model generalization
datagen = ImageDataGenerator(
    rescale=1.0/255,  # Normalize pixel values between 0 and 1
    rotation_range=20,  # Randomly rotate images by up to 20 degrees
    width_shift_range=0.2,  # Random horizontal shifts
    height_shift_range=0.2,  # Random vertical shifts
    shear_range=0.2,  # Shear transformations
    zoom_range=0.2,  # Random zooming
    horizontal_flip=True,  # Flip images horizontally
    validation_split=0.2  # Reserve 20% of data for validation
)

# Load training and validation datasets
train_generator = datagen.flow_from_directory(
    data_dir, target_size=img_size, batch_size=batch_size, class_mode='binary', subset='training'
)
val_generator = datagen.flow_from_directory(
    data_dir, target_size=img_size, batch_size=batch_size, class_mode='binary', subset='validation'
)

def build_cnn(optimizer='adam', dropout_rate=0.5):
    """Builds a simple CNN model with given optimizer and dropout rate."""
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),  # First convolutional layer
        MaxPooling2D(2,2),  # Pooling to reduce spatial dimensions
        
        Conv2D(64, (3,3), activation='relu'),  # Second convolutional layer
        MaxPooling2D(2,2),

        Conv2D(128, (3,3), activation='relu'),  # Third convolutional layer
        MaxPooling2D(2,2),
        
        Flatten(),  # Flatten feature maps into a single vector
        Dense(128, activation='relu'),  # Fully connected layer
        Dropout(dropout_rate),  # Dropout to reduce overfitting
        Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train CNN with different hyperparameters
optimizers = ['adam', 'sgd']  # Optimizer variations
dropout_values = [0.3, 0.5]  # Dropout variations

for opt in optimizers:
    for dropout in dropout_values:
        print(f"Training CNN with optimizer={opt}, dropout={dropout}")
        model = build_cnn(optimizer=opt, dropout_rate=dropout)

        # Callbacks: Stop training early if no improvement & adjust learning rate
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-5)
        ]

        # Train the model
        model.fit(train_generator, validation_data=val_generator, epochs=20, callbacks=callbacks, verbose=1)

        # Save the trained model
        model.save(f'face_mask_classifier_{opt}_{dropout}.keras')

# Feature extraction for ML models
def extract_features(generator):
    """Extracts features from images for SVM and MLP training."""
    features = []
    labels = []
    for batch_images, batch_labels in generator:
        batch_images = batch_images.reshape(batch_images.shape[0], -1)  # Flatten images
        features.append(batch_images)
        labels.append(batch_labels)
        if len(features) * batch_size >= generator.samples:
            break  # Stop after one full pass through data
    return np.vstack(features), np.concatenate(labels)

X_train, y_train = extract_features(train_generator)
X_val, y_val = extract_features(val_generator)


# Train SVM model
svm = SVC(kernel='linear')  # Using a linear kernel
svm.fit(X_train, y_train)
svm_preds = svm.predict(X_val)
svm_acc = accuracy_score(y_val, svm_preds)
print(f"SVM Validation Accuracy: {svm_acc:.4f}")

# Train Neural Network Classifier
mlp = MLPClassifier(hidden_layer_sizes=(128,), max_iter=500)  # Single hidden layer with 128 neurons
mlp.fit(X_train, y_train)
mlp_preds = mlp.predict(X_val)
mlp_acc = accuracy_score(y_val, mlp_preds)
print(f"Neural Network Validation Accuracy: {mlp_acc:.4f}")

# Final Accuracy Comparison
print("\n===== Final Accuracy Comparison =====")
print(f"SVM Accuracy: {svm_acc:.4f}")
print(f"Neural Network Accuracy: {mlp_acc:.4f}")

# Evaluate CNN models trained earlier
for opt in optimizers:
    for dropout in dropout_values:
        model = keras.models.load_model(f'face_mask_classifier_{opt}_{dropout}.keras')
        test_loss, test_acc = model.evaluate(val_generator)
        print(f"CNN (Optimizer={opt}, Dropout={dropout}) Accuracy: {test_acc:.4f}")
