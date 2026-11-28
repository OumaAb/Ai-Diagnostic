# Part 3 – Model Evaluation & Explainability

## 1. Overview

This part evaluates the trained DenseNet121 model for pneumonia classification using chest X-ray images.

The evaluation is performed on the independent test set using a classification threshold of **0.55**, selected previously on the validation set.

## 2. Test Dataset

- Total images: 880
- NORMAL: 239
- PNEUMONIA: 641
- Model: DenseNet121
- Image size: 224 × 224
- Classification threshold: 0.55

## 3. Evaluation Results

| Metric | Result |
|---|---:|
| Accuracy | 95.91% |
| Precision | 98.25% |
| Sensitivity | 96.10% |
| Specificity | 95.40% |
| F1-score | 97.16% |
| ROC-AUC | 99.20% |

### Confusion Matrix

- True Negatives (TN): 228
- False Positives (FP): 11
- False Negatives (FN): 25
- True Positives (TP): 616

## 4. Error Analysis

The model produced 36 classification errors:

- 11 False Positives: NORMAL images classified as PNEUMONIA.
- 25 False Negatives: PNEUMONIA images classified as NORMAL.

## 5. Explainability – Grad-CAM

Grad-CAM was applied to four representative test images:

- TN: correctly classified NORMAL image.
- TP: correctly classified PNEUMONIA image.
- FP: NORMAL image incorrectly classified as PNEUMONIA.
- FN: PNEUMONIA image incorrectly classified as NORMAL.

The heatmaps provide visual information about image regions contributing to the model's predictions. They do not establish clinical evidence of pneumonia.

## 6. Deliverables

- `test_metrics.csv`
- `test_predictions_threshold_055.csv`
- `test_errors_threshold_055.csv`
- `test_confusion_matrix.png`
- `test_roc_curve.png`
- `test_gradcam_overlay_comparison.png`

## 7. Important Note

The threshold was selected using the validation set and applied unchanged to the test set.

This project is an academic research prototype and is not intended for clinical diagnosis.
