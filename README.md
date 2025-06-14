# Deep-Learning-CT-Scan-Analysis
CT-Scan Chest Disease Classification with Deep Learning
This project employs Convolutional Neural Networks (CNNs) to classify chest CT-scan images into distinct categories: COVID-19, Normal (healthy lungs), and Viral Pneumonia. The primary goal is to develop a high-accuracy, robust, and interpretable deep learning model that assists medical professionals in rapid preliminary diagnosis of these critical lung conditions.

Project Overview
Topic: Image Classification for CT-Scan Chest Diseases

Frameworks: TensorFlow, Keras, NumPy, Matplotlib

Dataset: Chest CT-Scan Images on Kaggle

Test Accuracy Achieved: 95.75%

Use Case: Medical Diagnostics, specifically for the rapid identification and differentiation of lung pathologies from CT imagery.

Objectives
Develop and implement a robust Convolutional Neural Network (CNN) model for multiclass image classification of chest CT scans.

Analyze and visually compare the model's performance during training and validation using key metrics (accuracy and loss curves).

Investigate and demonstrate the effectiveness of data augmentation and regularization techniques (via Keras Callbacks) in improving model generalization and preventing overfitting.

Provide a clear, high-accuracy diagnostic tool with real-world applicability in medical imaging to aid clinicians.

Identify potential limitations of the current model and suggest areas for future improvements or extensions.

Model Architecture
The model is a sequential Convolutional Neural Network designed for image classification.

CNN Layers: Composed of multiple Conv2D and MaxPooling2D blocks for hierarchical feature extraction, followed by Flatten and Dense layers for classification.

Optimizer: Adam

Loss Function: Categorical Crossentropy

Metrics: Accuracy (monitored during training and evaluation)

Augmentation: Random rotations (rotation_range), zooms (zoom_range), horizontal flips (horizontal_flip), and brightness adjustments (brightness_range) applied to the training data.

Epochs: Trained for 50 epochs, with early stopping mechanisms.

Classes: 3 distinct classes (COVID, Normal, Viral Pneumonia).

Input Image Size: 128x128 pixels, 3 channels (RGB).

📊 Key Results
The model demonstrated strong performance on unseen data after training.

Metric

Score

Test Accuracy

95.75%

Test Loss

0.165

Learning Curves: Visualizations of training and validation accuracy/loss over epochs clearly illustrate the learning process and the point of optimal generalization (where early stopping would activate).

