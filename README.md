# Melanoma Detection using a Custom CNN

## 1. Introduction
This project aims to classify skin lesion images (from the ISIC dataset) into multiple classes (e.g., different skin cancer types, including melanoma). We use a Convolutional Neural Network (CNN) model built in TensorFlow/Keras to detect and classify these lesions.

## 2. Project Structure
- **melanoma_detection.ipynb**: Jupyter notebook containing all the code, from data reading to model training.
- **melanoma_detection.py**: Python script version of the same code (if required).
- **report.pdf**: A PDF document summarizing the findings, challenges, and conclusions.
- **data/**: (Optional) Folder containing the dataset if it’s not too large, or a note on how to download it.
- **README.md**: Current file describing the project.

## 3. Dataset
- **Dataset Source**: Skin cancer ISIC - The International Skin Imaging Collaboration (link if available).
- **Classes**: 9 classes representing various skin cancer types, including melanoma.
- **Folder Structure**:
  - `Train/` with subfolders for each class.
  - `Test/` with unseen images for final model evaluation.

## 4. Approach
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

8. **Evaluation & Results**  
   - Final accuracy, confusion matrix, and classification report on the validation set.  
   - Observations about how augmentation and oversampling impacted performance.

## 5. Findings
- **Baseline Model**: [Summary of accuracy/loss]
- **Augmented Model**: [Summary of changes in accuracy, any improvement in minority classes?]
- **Balanced Model**: [Summary of how oversampling influenced each class’s performance]

## 6. Conclusion
- **Overfitting or Underfitting**: Whether it was resolved by augmentation or not.  
- **Next Steps**: Potential improvements (e.g., hyperparameter tuning, deeper architectures, more sophisticated data balancing techniques).

## 7. Assumptions
- All images are RGB.  
- The dataset folders are structured as described: one folder per class in `Train/`.
