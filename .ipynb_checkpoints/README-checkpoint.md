# Breast Cancer Classification Using Machine Learning

## BITS ML Assignment-2

## A. Problem Statement

The objective of this project is to develop and compare multiple machine learning classification models for predicting whether a breast tumor is malignant or benign.

The project uses the Breast Cancer Wisconsin Diagnostic Dataset. Five classification models are trained and evaluated on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors Classifier
4. Gaussian Naive Bayes Classifier
5. Random Forest Classifier

The performance of each model is evaluated using the following metrics:

- Accuracy
- Area Under the ROC Curve (AUC)
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

An interactive Streamlit web application is developed to allow users to:

- Upload test data in CSV format
- Select a machine learning model
- Generate predictions on uploaded test data
- View model evaluation metrics
- Compare the performance of all implemented models
- Review a classification report
- Visualize the confusion matrix

---

## B. Dataset Description

### Dataset Name

Breast Cancer Wisconsin Diagnostic Dataset

### Dataset Source

Scikit-Learn Built-in Dataset

### Dataset Characteristics

- Total Instances: 569
- Total Features: 30
- Target Classes: 2

### Target Variable

- 0 = Malignant Tumor
- 1 = Benign Tumor

### Feature Description

The dataset contains numerical features computed from digitized images of breast mass cell nuclei. These measurements include:

- Radius
- Texture
- Perimeter
- Area
- Smoothness
- Compactness
- Concavity
- Symmetry
- Fractal Dimension

and several related statistical measurements.

### Why This Dataset Was Selected

The Breast Cancer Wisconsin Dataset satisfies the assignment requirements because:

- It is a classification dataset.
- It contains more than 500 instances.
- It contains more than 12 features.
- It is suitable for comparing multiple classification algorithms.
- It is widely used for machine learning benchmarking and evaluation.