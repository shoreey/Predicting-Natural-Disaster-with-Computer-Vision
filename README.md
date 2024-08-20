# Natural Disaster Prediction with Computer Vision

## Project Overview

This project aims to predict natural disasters using computer vision techniques. It utilizes a dataset of images categorized into four types of natural disasters: Cyclones, Wildfires, Floods, and Earthquakes. The project employs transfer learning with pre-trained models such as ResNet50, VGG16, and MobileNet to classify disaster images.

## Dataset

The dataset is stored in the directory `basedir:\DATASET\Cyclone_Wildfire_Flood_Earthquake_Database`. It contains images of four types of natural disasters:

1. Cyclones
2. Wildfires
3. Floods
4. Earthquakes

## Code Structure

The main script performs the following tasks:

1. Loads and preprocesses the image data.
2. Implements transfer learning using pre-trained models (ResNet50, VGG16, MobileNet).
3. Performs hyperparameter tuning with different learning rates and batch sizes.
4. Trains and evaluates models.
5. Generates confusion matrices and performance plots.

## Dependencies

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Usage

1. Ensure you have the required dependencies installed.
2. Update the `base_dir` variable to point to your dataset directory.
3. Run the script to train models, evaluate performance, and visualize results.

## Model Architecture

The project uses transfer learning with the following pre-trained models:

- ResNet50
- VGG16
- MobileNet

These models are fine-tuned on our specific dataset for natural disaster classification.

## Data Preprocessing

- Images are rescaled and resized to 224x224 pixels.
- Data augmentation techniques are applied, including rotation, shifting, shearing, zooming, and flipping.

## Training

- Validation split: 20%
- Test split: 10%
- Hyperparameters tuned:
  - Learning rates: [0.001, 0.0001]
  - Batch sizes: [32, 64]
- Epochs: 10

## Results

The best performance for each model:

1. MobileNet: 88% accuracy
2. VGG16: 82% accuracy
3. ResNet50: 65% accuracy

MobileNet demonstrated the highest accuracy in classifying natural disaster images.

## Visualizations

The script generates the following visualizations for each model configuration:

1. Confusion Matrix
2. Training and Validation Loss
3. Training and Validation Accuracy
4. Comparison bar plot of model accuracies

## Future Work

- Fine-tune hyperparameters further to improve performance.
- Experiment with ensemble methods to combine model predictions.
- Collect more diverse data to improve generalization.
- Implement real-time prediction on video streams.
