# CNN for 2D Landmark Prediction Using AFLW2K3D Dataset

## Description
This project uses a Convolutional Neural Network (CNN) to predict 2D facial landmarks from images in the AFLW2K3D dataset. The AFLW2K3D dataset provides 68 3D facial landmarks, but only the x and y coordinates (2D) are used for this project. The model predicts 68 (x, y) landmarks for each image, flattening the array into a 1D vector of size 136.

## Features
- **Data Preprocessing**: Extracts and reshapes the landmark data from the AFLW2K3D dataset.
- **Convolutional Neural Network (CNN)**: A CNN is built and trained to predict facial landmarks.
- **Training and Validation**: The model is trained and validated on the AFLW2K3D dataset, and loss metrics are visualized.

## Requirements
- Python 3.x
- TensorFlow
- NumPy
- Matplotlib
- TensorFlow Datasets (tfds)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/amfathy/CNN-2D-Landmark-Prediction-Using-AFLW2K3D-Dataset.git
