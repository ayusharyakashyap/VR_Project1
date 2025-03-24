import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import gc
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
# Configure TensorFlow for CPU optimization
tf.config.threading.set_inter_op_parallelism_threads(4)  # Adjust based on CPU cores
tf.config.threading.set_intra_op_parallelism_threads(4)  # Adjust based on CPU cores

# Enable Intel oneDNN optimizations if available
try:
    tf.config.optimizer.set_jit(True)
    print("Intel oneDNN optimizations enabled")
except:
    print("Intel oneDNN optimizations not available")

print("Using CPU for computation")

def load_data(face_crop_dir, face_crop_seg_dir):
    """Load all images and masks at once."""
    image_files = sorted(os.listdir(face_crop_dir))
    images = []
    masks = []
    
    for img_file in image_files:
        img_path = os.path.join(face_crop_dir, img_file)
        mask_path = os.path.join(face_crop_seg_dir, img_file)
        
        if not os.path.exists(mask_path):
            continue
            
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None or mask is None:
            continue
            
        img = cv2.resize(img, (128, 128))
        mask = cv2.resize(mask, (128, 128))
        _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
        
        images.append(img)
        masks.append(mask)
    
    return np.array(images), np.array(masks)

def apply_traditional_segmentation(images):
    """Apply traditional segmentation techniques to detect face masks."""
    results = []
    
    for img in images:
        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for masks
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 30, 255])
        
        # Create color masks
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(blue_mask, white_mask)
        
        # Apply morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Normalize to binary
        mask = (mask > 0).astype(np.uint8)
        
        results.append(mask)
    
    return np.array(results)

def build_unet(input_size=(128, 128, 3)):
    """Build U-Net model architecture."""
    inputs = Input(input_size)
    
    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # Bridge
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    drop4 = Dropout(0.5)(conv4)
    
    # Decoder
    up5 = Conv2D(256, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(drop4))
    merge5 = concatenate([conv3, up5], axis=3)
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(merge5)
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(conv5)
    
    up6 = Conv2D(128, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv5))
    merge6 = concatenate([conv2, up6], axis=3)
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(conv6)
    
    up7 = Conv2D(64, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv1, up7], axis=3)
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(conv7)
    
    outputs = Conv2D(1, 1, activation='sigmoid')(conv7)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def train_unet(images, masks, epochs=30, batch_size=8, validation_split=0.2):
    """Train U-Net model with direct memory management."""
    # Build model
    model = build_unet(input_size=(128, 128, 3))
    
    # Normalize and prepare data
    images = images.astype('float32') / 255.0
    masks = np.expand_dims(masks, axis=-1).astype('float32')
    
    # Split data
    num_samples = len(images)
    num_train = int(num_samples * (1 - validation_split))
    
    train_images = images[:num_train]
    train_masks = masks[:num_train]
    val_images = images[num_train:]
    val_masks = masks[num_train:]
    
    # Callbacks with memory cleanup
    class MemoryCleanupCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            gc.collect()
            tf.keras.backend.clear_session()
    
    callbacks = [
        ModelCheckpoint('unet_mask_segmentation.h5', save_best_only=True, monitor='val_loss'),
        EarlyStopping(patience=10, monitor='val_loss'),
        MemoryCleanupCallback()
    ]
    
    # Train with memory monitoring
    print(f"Training U-Net model...")
    try:
        history = model.fit(
            train_images,
            train_masks,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(val_images, val_masks),
            callbacks=callbacks
        )
    except tf.errors.ResourceExhaustedError:
        print("Memory exhausted. Try reducing batch size or image dimensions.")
        return None
    
    model.load_weights('unet_mask_segmentation.h5')
    return model

def predict_masks(model, images):
    """Predict segmentation masks using trained U-Net model."""
    # Normalize images
    normalized_images = images.astype('float32') / 255.0
    
    # Predict
    predicted_masks = model.predict(normalized_images)
    
    # Convert to binary
    binary_masks = (predicted_masks > 0.5).astype(np.uint8)
    binary_masks = np.squeeze(binary_masks, axis=-1)
    
    return binary_masks

def calculate_metrics(predicted_masks, ground_truth_masks):
    """Calculate evaluation metrics for segmentation results."""
    metrics = {}
    
    # Ensure binary masks
    predicted = (predicted_masks > 0).astype(np.uint8)
    ground_truth = (ground_truth_masks > 0).astype(np.uint8)
    
    # Calculate metrics for each image
    num_images = len(predicted)
    iou_scores = []
    dice_scores = []
    
    for i in range(num_images):
        # Intersection over Union (IoU)
        intersection = np.logical_and(predicted[i], ground_truth[i])
        union = np.logical_or(predicted[i], ground_truth[i])
        iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
        iou_scores.append(iou)
        
        # Dice coefficient
        dice = 2 * np.sum(intersection) / (np.sum(predicted[i]) + np.sum(ground_truth[i])) \
            if (np.sum(predicted[i]) + np.sum(ground_truth[i])) > 0 else 0
        dice_scores.append(dice)
    
    # Average metrics
    metrics['mean_iou'] = np.mean(iou_scores)
    metrics['mean_dice'] = np.mean(dice_scores)
    
    return metrics

def visualize_results(images, masks, traditional_results, unet_results, num_samples=3):
    """Visualize and compare segmentation results."""
    indices = np.random.choice(len(images), min(num_samples, len(images)), replace=False)
    
    for i, idx in enumerate(indices):
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(1, 4, 1)
        plt.imshow(cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        
        # Ground truth mask
        plt.subplot(1, 4, 2)
        plt.imshow(masks[idx], cmap='gray')
        plt.title('Ground Truth Mask')
        plt.axis('off')
        
        # Traditional segmentation
        plt.subplot(1, 4, 3)
        plt.imshow(traditional_results[idx], cmap='gray')
        plt.title('Traditional Segmentation')
        plt.axis('off')
        
        # U-Net segmentation
        plt.subplot(1, 4, 4)
        plt.imshow(unet_results[idx], cmap='gray')
        plt.title('U-Net Segmentation')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'result_comparison_{i}.png')
        plt.show()

def main():
    # Set memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    # Set paths
    face_crop_dir = os.path.join('MSFD', 'MSFD', '1', 'face_crop')
    face_crop_seg_dir = os.path.join('MSFD', 'MSFD', '1', 'face_crop_segmentation')
    
    # Load all data at once
    print("Loading data...")
    images, masks = load_data(face_crop_dir, face_crop_seg_dir)
    
    # Traditional segmentation
    print("\nApplying traditional segmentation techniques...")
    traditional_results = apply_traditional_segmentation(images)
    traditional_metrics = calculate_metrics(traditional_results, masks)
    
    # Clean up memory
    gc.collect()
    
    print(f"Traditional Segmentation Metrics: {traditional_metrics}")
    
    # U-Net segmentation
    print("\nTraining U-Net for mask segmentation...")
    model = train_unet(images, masks, epochs=1, batch_size=8)
    
    if model is None:
        print("Training failed due to memory constraints.")
        return
    
    # Predict and evaluate
    print("Predicting masks using U-Net...")
    unet_results = predict_masks(model, images)
    unet_metrics = calculate_metrics(unet_results, masks)
    
    # Clean up memory
    gc.collect()
    
    print(f"U-Net Segmentation Metrics: {unet_metrics}")
    
    # Compare results
    print("\nComparison of Traditional vs U-Net Segmentation:")
    print(f"Traditional: {traditional_metrics}")
    print(f"U-Net: {unet_metrics}")
    
    # Visualize results with memory cleanup
    visualize_results(
        images[-min(3, len(images)):],
        masks[-min(3, len(masks)):],
        traditional_results[-min(3, len(traditional_results)):],
        unet_results[-min(3, len(unet_results)):]
    )
    gc.collect()

if __name__ == '__main__':
    main()