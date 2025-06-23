
# 🧬 Methodology: Liver Tumor Classification with CNN

This project aims to develop a deep learning model using a **Convolutional Neural Network (CNN)** to classify liver ultrasound images into three categories: **Benign**, **Malignant**, and **Normal**.

---

## 1. 🧹 Data Preparation

- **Dataset Structure**: Images were organised into class-labeled folders (`train/`, `val/`, `test/`) with subfolders for each class (Benign, Malignant, Normal).
- **Transformations**: Applied using `torchvision.transforms`, including resizing, normalisation, and tensor conversion.
- **Class Imbalance**: More examples of *Malignant* class were observed, impacting performance distribution.

---

## 2. 🧠 Model Architecture

Built from scratch using PyTorch:
- **Layers**:
  - Convolutional + ReLU
  - MaxPooling
  - Dropout
  - Fully Connected layers
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam

---

## 3. 🏋️ Training & Evaluation

- Trained over multiple epochs with monitoring of validation accuracy/loss.
- Manual early stopping applied.
- Metrics evaluated on a separate **test set**.

---

## 4. 📊 Test Performance

### Classification Report:

| Class         | Precision | Recall | F1-Score |
|---------------|-----------|--------|----------|
| **Benign**    | 0.61      | 0.57   | 0.59     |
| **Malignant** | 0.85      | 0.91   | 0.88     |
| **Normal**    | 0.81      | 0.65   | 0.72     |

- **Accuracy**: 78%
- **Weighted F1 Score**: 0.7782

🧠 Malignant detection is strong — clinically important. Benign cases showed more misclassifications, likely from data imbalance or class overlap.

---

## 5. 🎯 Key Takeaways

- The CNN achieved good accuracy (for a dataset this small) in the context of a 3-class medical classification task.
- Strong recall on Malignant — good for real-world early detection.
- Evaluation metrics and interpretability were crucial for assessing fairness across classes.

> 🔗 Built with PyTorch | Metrics via scikit-learn | Dataset: Liver ultrasound images

---

## 6. 🔍 Experiments & Iterations

### 🔁 k-Fold Cross Validation and Transforms
One of the first adjustments I made to tackle the small size of the dataset was to perform transformations in my transform step such as `RandomHorizontalFlip`, `RandomRotation` and `RandomResizedCrop`. This had the opposite effect as expected due to training and validation performance slightly decreasing.

I implemented 5-fold cross validation using stratified sampling to improve generalisation. However, training time increased significantly and accuracy gains were negligible (≈+1%), so I reverted to hold-out validation for simplicity and speed.

### 🧪 Original Architecture
The first model had:
- 2 convolutional layers
- No dropout
- Final layer used Sigmoid (binary assumption)

It underperformed due to overfitting and incorrect output shape. Adjusted to 3-class Softmax classifier with added Dropout.

### 🎯 Hyperparameter Tuning
I explored:
- **Learning rates**: 0.1 to 0.00001 — best results at 0.0005
- **Batch sizes**: 16 vs 32 — 32 was faster with stable gradients
- **Dropout**: 0.2 vs 0.3 in FC layer to reduce overfitting
- **Epochs**: Experimented in increments of 5 up to 35 epochs

These optimisations led to a ~10% boost in F1 score on the validation set.
