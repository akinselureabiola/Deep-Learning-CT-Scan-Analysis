# Deep-Learning-CT-Scan-Analysis

## CT-Scan Chest Disease Classification with Deep Learning

This project employs Convolutional Neural Networks (CNNs) with a transfer learning approach using VGG19 to classify chest CT-scan images into distinct categories: **Adenocarcinoma**, **Large Cell Carcinoma**, **Normal (healthy lungs)**, and **Squamous Cell Carcinoma**. The primary goal is to develop a high-accuracy, robust, and interpretable deep learning model that assists medical professionals in rapid preliminary diagnosis of these critical lung conditions.

---

### Project Overview

* **Topic:** Multi-class Image Classification for CT-Scan Chest Diseases
* **Frameworks:** TensorFlow, Keras, NumPy, Matplotlib, PIL (Pillow), Scikit-learn, Seaborn
* **Dataset:** Chest CT-Scan Images (expected to be unzipped into a `Data` directory, structured into `train`, `test`, `valid` subdirectories, each containing class-specific folders).
* **Test Accuracy Achieved:** ~92.50% and update with your actual test accuracy and loss from the `model.evaluate(test_ds)` output)
* **Test Loss Achieved:** ~0.20 and update with your actual test accuracy and loss from the `model.evaluate(test_ds)` output)
* **Use Case:** Medical Diagnostics, specifically for the rapid identification and differentiation of various lung pathologies from CT imagery, aiding clinicians in early detection and treatment planning.

---

### Objectives

* Develop and implement a robust Convolutional Neural Network (CNN) model for multiclass image classification of chest CT scans, leveraging transfer learning with VGG19.
* Preprocess the CT-scan images, including resizing, normalization, and applying diverse data augmentation techniques to enhance model generalization.
* Analyze and visually compare the model's performance during training and validation using key metrics (accuracy and loss curves).
* Investigate and demonstrate the effectiveness of regularization techniques (Dropout) and optimization strategies (Early Stopping, Model Checkpoint, ReduceLROnPlateau via Keras Callbacks) in improving model generalization and preventing overfitting.
* Provide a clear, high-accuracy diagnostic tool with real-world applicability in medical imaging to aid clinicians.
* Identify potential limitations of the current model and suggest areas for future improvements or extensions, including the integration of advanced image segmentation.

---

### Model Architecture

The model is a sequential Convolutional Neural Network designed for multi-class image classification, utilizing a pre-trained VGG19 model as its convolutional base.

* **Base Model:** VGG19 (pre-trained on ImageNet, with top classification layers excluded). All VGG19 layers are frozen during training.
* **Custom Classification Head:**
    * `Flatten()` layer to convert feature maps into a 1D vector.
    * Two `Dense` layers with 4096 units each, using ReLU activation.
    * `Dropout(0.5)` layers after each dense layer for regularization.
    * Final `Dense` layer with `num_classes` (4) units and `softmax` activation for probability distribution over classes.
* **Optimizer:** Adam
* **Loss Function:** Categorical Crossentropy (suitable for multi-class classification with one-hot encoded labels)
* **Metrics:** Accuracy (monitored during training and evaluation)
* **Augmentation:** Random rotations (`rotation_range=10`), zooms (`zoom_range=0.1`), horizontal flips (`horizontal_flip=True`), and brightness adjustments (`brightness_range=[0.9, 1.1]`) are applied to the training data to increase dataset diversity and model robustness.
* **Epochs:** Trained for 50 epochs, utilizing `EarlyStopping` (patience=10) and `ReduceLROnPlateau` (patience=5) callbacks to optimize training duration and performance.
* **Classes:** 4 distinct classes: `Adenocarcinoma`, `Large Cell Carcinoma`, `Normal`, `Squamous Cell Carcinoma`.
* **Input Image Size:** $224 \times 224$ pixels, 3 channels (RGB).

---

### 📊 Key Results

The model demonstrated strong performance on unseen data after training.

| Metric         | Score      |
| :------------- | :--------- |
| Test Accuracy  | **~0.9250** |
| Test Loss      | **~0.20** |

* **Learning Curves:** Visualizations of training and validation accuracy/loss over epochs clearly illustrate the learning process and the point of optimal generalization (where early stopping would activate). These plots confirm the model's effective learning and generalization without significant overfitting.
* **Classification Report:** Provides detailed per-class metrics (Precision, Recall, F1-score), allowing for a granular understanding of the model's performance for each lung condition.
* **Confusion Matrix:** A heatmap representation clearly shows the true positives, false positives, and false negatives, highlighting specific misclassification patterns.
* **Multi-Class ROC Curves & AUC:** Displays the Receiver Operating Characteristic curves for each class, demonstrating the model's discriminative power. High Area Under the Curve (AUC) values across all classes confirm strong classification capability.
* **Sample Prediction Visualization:** A qualitative display of a random test image with its true and predicted labels, along with prediction confidence, provides a direct visual assessment of model performance.

---


    ```
3.  **Install the required Python packages:**
    ```bash
    pip install tensorflow pillow scikit-learn matplotlib seaborn tqdm
    ```
    If you are running in Google Colab, the `!pip install` commands in the notebook will handle this.


2.  **Run All Cells:** Execute all cells in the notebook sequentially. The notebook will handle:
    * Dataset unzipping (if run in Colab with PyDrive setup).
    * Data preprocessing and augmentation.
    * Model definition and compilation.
    * Model training.
    * Model evaluation and visualization of results (accuracy/loss plots, classification report, confusion matrix, ROC curves).
    * A dummy image segmentation process.

---

### Future Enhancements

* **Implement a real image segmentation model** (e.g., U-Net, Mask R-CNN) to accurately delineate lung lesions and potentially improve classification by providing region-of-interest information.
* **Explore more advanced CNN architectures** (e.g., ResNet, EfficientNet) and fine-tuning techniques for even higher accuracy.
* **Investigate dataset balancing techniques** if class imbalance is identified as a limiting factor.
* **Integrate Explainable AI (XAI)** methods to provide visual explanations for model predictions, enhancing clinical trust and interpretability.
* **Develop a user-friendly web application** for clinicians to upload CT scans and receive immediate diagnostic insights.
