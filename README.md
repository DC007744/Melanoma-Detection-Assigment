# Melanoma Detection using a Custom CNN

## 1. Introduction
This project aims to classify skin lesion images (from the ISIC dataset) into multiple classes (e.g., different skin cancer types, including melanoma). We use a Convolutional Neural Network (CNN) model built in TensorFlow/Keras to detect and classify these lesions.

## 2. Dataset
- **Dataset Source**: Skin cancer ISIC - The International Skin Imaging Collaboration.
- **Classes**: 9 classes representing various skin cancer types, including melanoma.
- **Folder Structure**:
  - `Train/` with subfolders for each class.
  - `Test/` with unseen images for final model evaluation.

## 3. Approach
1. **Data Reading & Understanding**  
   - Paths defined for `train_dir` and `test_dir`.  
   - Basic stats on class distribution.

2. **Dataset Creation**  
   - Used `ImageDataGenerator` to load images in batches of 32, resized to 180×180 pixels.  
   - 80-20 split for train-validation.

3. **Visualization**  
   - Displayed sample images (one per class) to verify correctness.

4. **Model Architecture**  
   - Simple CNN with 3 convolution blocks, each followed by pooling.  
   - Dense layer of size 512, dropout of 0.5, and a final softmax layer.

5. **Training (Phase 1)**  
   - Trained for 20 epochs using `adam` and `categorical_crossentropy`.  
   - Monitored training vs validation accuracy and loss to detect signs of over/underfitting.

6. **Data Augmentation**  
   - Added random rotations, shifts, shear, zoom, and flips to improve model robustness.

7. **Handling Class Imbalance**  
   - Used `RandomOverSampler` to oversample minority classes.  
   - Retrained the same CNN and compared performance metrics.
