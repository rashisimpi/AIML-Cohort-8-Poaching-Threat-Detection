# Audio Classification for Environmental Monitoring

## Project Overview

This repository contains the final stage of an **audio classification project** focused on identifying environmental sounds related to wildlife activity and potential threats such as **illegal logging**.

The goal of this stage is to **evaluate trained machine learning models**, validate their reliability using **cross validation**, and recommend the most suitable model for deployment.

### Machine Learning Models Evaluated

* Random Forest Classifier (RFC)
* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)

The evaluation process includes:

* Performance metrics
* Confusion matrices
* ROC curves
* Cross-validation analysis

These techniques ensure a **reliable comparison between models**.

---

# Dataset

The dataset used for this project:

```
audio_features.csv
```

This dataset contains **audio feature vectors extracted from environmental recordings** along with encoded class labels.

### Input Features

Numerical audio features extracted from recordings such as:

* MFCC features
* Spectral features
* Zero Crossing Rate
* Other acoustic descriptors

### Target Variable

```
label_encoded
```

A numerical label representing different environmental sound categories.

---

# Libraries Used

The following Python libraries were used:

```python
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt
import seaborn as sns
```

---

# Model Evaluation

Each classifier was evaluated using several **performance metrics** to analyze how well the models detect sound events.

### Evaluation Metrics

**Accuracy**
Measures the overall proportion of correct predictions.

**Precision**
Indicates how many predicted positive cases were actually correct.

**Recall (Sensitivity)**
Measures the model’s ability to detect actual positive cases.

**F1 Score**
Harmonic mean of precision and recall.

**False Positive Rate (FPR)**
Shows how often normal events are incorrectly classified as threats.

**False Negative Rate (FNR)**
Indicates how often actual threats are missed.

**Classification Report**
Provides detailed per-class precision, recall, and F1 scores.

---

# Confusion Matrix Analysis

A confusion matrix heatmap was generated for each model to visualize:

* Correct predictions
* Incorrect predictions
* Misclassified classes

This helps identify which sound categories the model struggles to classify.

---

# ROC Curve Analysis

ROC curves were generated for:

* Random Forest
* KNN
* SVM

The ROC curve shows the relationship between:

* True Positive Rate (Sensitivity)
* False Positive Rate

The **Area Under the Curve (AUC)** was calculated for each model.

A **combined ROC curve comparison graph** was also generated.

---

# Cross Validation

To ensure models perform consistently on unseen data, **Stratified K-Fold Cross Validation** was applied.

### Method Used

```
StratifiedKFold (5 folds)
```

This ensures:

* Each fold maintains the same class distribution
* Evaluation results remain balanced and reliable

Cross-validation scores were computed for:

* Random Forest
* KNN (with feature scaling)
* SVM (with feature scaling)

---

# Model Comparison

All models were compared using the following metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* False Positive Rate
* False Negative Rate
* Cross Validation Accuracy
* ROC AUC Score

This comparison helps determine the **best performing classifier**.

---

# Model Recommendation

Based on:

* Evaluation metrics
* Cross validation scores
* Confusion matrix analysis
* ROC curve comparison

the **most reliable model** was recommended for deployment.

The selected model provides the best balance between:

* Accurate sound classification
* Low false alarm rate
* Minimal missed threat detection

This is important for **environmental monitoring systems**.

---

# Project Team Contributions

### Rashi — Model Evaluation (Random Forest)

Evaluated the Random Forest classifier and calculated metrics such as accuracy, precision, recall, F1 score, false positive rate, and false negative rate. Generated confusion matrices and ROC curves.

### Reethu — Model Evaluation (KNN & SVM)

Evaluated the KNN and SVM models by calculating performance metrics and generating classification reports. Created confusion matrices and ROC curves.

### Bharath — Cross Validation

Implemented Stratified K-Fold Cross Validation to test model reliability across multiple data splits.

### Poojith — Model Recommendation

Performed final model comparison using evaluation metrics and cross validation results. Recommended the most suitable classifier for deployment.

---

# Project Folder Structure

```
project/
│
├── data/
│   └── audio_features.csv
│
├── notebooks/
│   └── model_evaluation.ipynb
│
├── results/
│   ├── confusion_matrix.png
│   └── roc_curve.png
│
├── README.md
└── requirements.txt
```

---

# How to Run the Project

### 1 Install Required Libraries

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 2 Prepare the Dataset

Place the dataset file:

```
audio_features.csv
```

inside the **data folder**.

### 3 Run the Notebook

You can run the project using:

* Google Colab
* Jupyter Notebook
* VS Code

Run the notebook cells in this order:

1. Import libraries
2. Load dataset
3. Model development
4. Model evaluation
5. Cross validation
6. Model recommendation

---

# Output Generated

### Performance Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* False Positive Rate
* False Negative Rate
* Classification Reports

### Visualizations

* Confusion matrices for all models
* Individual ROC curves
* Combined ROC comparison graph

### Validation Results

* 5-fold cross validation scores for each classifier

---

# Final Outcome

This project provides a **complete evaluation and comparison of machine learning classifiers for environmental sound classification**.

By combining **performance metrics, visualization, and cross validation**, the project identifies the **most reliable model for detecting wildlife sounds and environmental threats**.
