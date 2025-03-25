# Face Mask Detection using CNN & ML Classifiers

## Introduction
Face mask detection is a crucial application in ensuring public safety, especially in the wake of global health crises. This project consists of two tasks:

- **Task A:** Detecting whether a person is wearing a face mask or not using machine learning techniques. The objective is to classify images into two categories: *with_mask* and *without_mask* using feature extraction methods such as *HOG (Histogram of Oriented Gradients), Sobel Edge Detection, and Canny Edge Detection*. The extracted features are used to train classifiers like **Support Vector Machine (SVM)** and **Multi-Layer Perceptron (MLP)**.

- **Task B:** Implementing a face mask detection system using a **Convolutional Neural Network (CNN)** alongside traditional machine learning classifiers such as **SVM** and **MLP**. The objective is to compare the performance of these models in detecting whether a person is wearing a mask or not.

## Dataset
The dataset used for training and evaluation is stored in the `Face-Mask-Detection/dataset` directory. It consists of images labeled as either "with mask" or "without mask." The dataset is preprocessed by resizing images to 128x128 pixels and applying data augmentation techniques for improved generalization.

### 🔹 Structure of Dataset

```
/dataset
│── with_mask/        # Images of people wearing masks
│── without_mask/     # Images of people without masks
```

Each category contains multiple grayscale images of size *64x64 pixels*.

## Task A: Machine Learning-Based Face Mask Detection

### Methodology
#### Steps Involved
1. **Data Preprocessing**
   - Convert images to grayscale.
   - Resize all images to 64x64 pixels for consistency.
   
2. **Feature Extraction**
   - *HOG (Histogram of Oriented Gradients)* for texture information.
   - *Sobel Edge Detection* to capture edge-based features.
   - *Canny Edge Detection* to extract prominent edges.
   - Concatenate all extracted features into a single feature vector.

3. **Data Splitting & Standardization**
   - Split the dataset into *80% training* and *20% testing*.
   - Standardize features using *StandardScaler* to normalize the data.

4. **Model Training**
   - Train a *Support Vector Machine (SVM)* classifier.
   - Train a *Multi-Layer Perceptron (MLP)* neural network.

5. **Model Evaluation**
   - Use *accuracy score* and *classification report* to evaluate performance.

### Machine Learning Classifiers Performance
| Model  | Validation Accuracy |
|--------|---------------------|
| SVM    | 85.83%               |
| MLP    | 90.59%               |

### Hyperparameters & Experiments
####  SVM Model
- Kernel: **linear**
- Default parameters used

#### MLP Model
- Hidden Layers: **(100,)**
- Activation: **ReLU**
- Max Iterations: **500**

#### Variations Tried
- Different hidden layer sizes for MLP
- Different pixel-per-cell values for HOG
- Feature selection methods to improve performance

### Observations & Analysis
- *Feature Engineering Impact:* Combining HOG, Sobel, and Canny features improved classification accuracy.
- *Challenges Faced:*
  - Dataset imbalance (mitigated by data augmentation techniques).
  - Feature extraction tuning required extensive experimentation.
- *Potential Improvements:*
  - Implement *CNN-based models* for improved performance.
  - Use *U-Net* for segmentation-based classification.

## Task B: Deep Learning-Based Face Mask Detection

### Methodology
1. **Data Preprocessing:**
   - Images resized to 128x128.
   - Data augmentation: Rescaling, rotation, width/height shift, shear, zoom, and horizontal flipping.
   - Train-validation split: 80%-20%.
2. **CNN Training:**
   - Different dropout values (0.3, 0.5) and optimizers (Adam, SGD) tested.
   - Binary cross-entropy used as the loss function.
   - Callbacks: EarlyStopping, ReduceLROnPlateau.

### CNN Model Architecture
- **Input:** 128x128 RGB images
- **Layers:**
  - Conv2D (32 filters, 3x3, ReLU) + MaxPooling (2x2)
  - Conv2D (64 filters, 3x3, ReLU) + MaxPooling (2x2)
  - Conv2D (128 filters, 3x3, ReLU) + MaxPooling (2x2)
  - Flatten + Dense (128, ReLU)
  - Dropout (0.3 or 0.5 for regularization)
  - Dense (1, Sigmoid for binary classification)
- **Optimizers tested:** Adam, SGD
- **Loss Function:** Binary Cross-Entropy

### CNN Performance Across Hyperparameter Variations
| Optimizer | Dropout | Validation Accuracy |
|-----------|---------|---------------------|
| Adam      | 0.3     | **95.2%**           |
| Adam      | 0.5     | **94.7%**           |
| SGD       | 0.3     | 91.5%               |
| SGD       | 0.5     | 90.2%               |

### Machine Learning Classifiers Performance
| Model  | Validation Accuracy |
|--------|---------------------|
| SVM    | 86.4%               |
| MLP    | 88.7%               |

## Evaluation Metrics
The models were evaluated using the following metrics:
- **Accuracy:** Measures the proportion of correctly classified images.
- **IoU (Intersection over Union):** Evaluates overlap between predicted and ground truth masks (for segmentation models).
- **Dice Score:** Measures similarity between prediction and ground truth.

## Observations and Analysis
- The **CNN model (Adam, Dropout=0.3)** achieved the highest accuracy of **95.2%**, outperforming both the SVM and MLP classifiers.
- The **MLP classifier (88.7%)** performed better than the **SVM (86.4%)**, but both were inferior to CNN models.
- The choice of **optimizer and dropout rate** significantly impacts CNN performance, with **Adam** generally outperforming **SGD**.
- Machine learning models, although simpler, struggle to capture the spatial features as effectively as CNNs.

## How to Run the Code

### Prerequisites
Ensure you have the following dependencies installed:
```bash
pip install numpy opencv-python matplotlib scikit-learn scikit-image tensorflow keras
```

### Steps to Execute
1. Clone the repository into your working directory:
   ```bash
   git clone https://github.com/chandrikadeb7/Face-Mask-Detection.git
   ```
2. The required dataset would be inside the `Face-Mask-Detection/dataset/` folder.
3. Run the script to execute both Task A and Task B:
   ```bash
   python main.py  # Runs Task A and Task B
   ```

### Expected Outputs:
- **Task A:** Model evaluation results, including accuracy for **SVM (86.4%)** and **MLP (88.7%)**.
- **Task B:** CNN training results, with the best accuracy achieved using **Adam (95.2%)**.

The trained CNN models will be saved in the working directory with filenames like:
```bash
face_mask_classifier_adam_0.3.keras
```

## Future Improvements
- Implementing transfer learning with pre-trained models like MobileNetV2 or ResNet.
- Enhancing dataset size for improved generalization.
- Fine-tuning hyperparameters further with techniques like Bayesian optimization.
