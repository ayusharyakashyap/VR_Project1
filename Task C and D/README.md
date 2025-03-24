# Face Mask Segmentation Project

This project implements and compares traditional image segmentation techniques with U-Net deep learning-based segmentation for face mask detection. It provides a comprehensive analysis of both approaches, demonstrating their effectiveness in identifying and segmenting face masks in images.

## Project Overview

The project uses two main approaches for face mask segmentation:
1. Traditional Image Processing: Uses HSV color space and morphological operations
2. Deep Learning: Implements U-Net architecture for semantic segmentation

## Project Structure

```
├── face_mask_segmentation.py   # Main implementation script
├── requirements.txt            # Python dependencies
├── requirements_tensorflow.txt  # TensorFlow specific requirements
├── MSFD/                      # Dataset directory
│   └── MSFD/
│       └── 1/                 # Dataset directory
│           ├── face_crop/              # Image dataset
│           └── face_crop_segmentation/ # Ground truth masks
└── unet_mask_segmentation.h5   # Trained U-Net model weights
```

## Setup and Installation

1. Clone this repository to your local machine

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements_tensorflow.txt
   ```

## Running the Project

To run the face mask segmentation comparison:

```bash
python face_mask_segmentation.py
```

This will:
1. Load the face mask dataset
2. Apply traditional segmentation techniques
3. Train and apply U-Net based segmentation
4. Generate comparison metrics and visualizations

## Output

The script generates several output files:
- `result_comparison_*.png`: Visual comparison of segmentation results
- `unet_vs_traditional_seg_iou_dice.png`: Performance metrics comparison
- `unet_mask_segmentation.h5`: Trained U-Net model weights

## Requirements

- Python 3.7+
- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib

For detailed requirements, see `requirements.txt` and `requirements_tensorflow.txt`.