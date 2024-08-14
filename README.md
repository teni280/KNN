# K-nearest neighbours (KNN) algorithm 

### Overview

This notebook demonstrates a simple machine learning pipeline using the k-Nearest Neighbors (kNN) algorithm to classify a dataset containing information about cars. The dataset is loaded, processed, and used to train the model. After training, the notebook evaluates the performance of the model using common classification metrics.

### Project Structure
- **Import Libraries**: Necessary Python libraries such as `pandas` (for data manipulation), `sklearn` (for model building and evaluation), and `numpy` (for numerical computations) are imported.
  
- **Data Loading and Preprocessing**: 
  - The car dataset is loaded from a CSV file named `car.data`.
  - The dataset has no headers, so they are manually assigned: `['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']`.
  - Label encoding is applied to transform categorical variables into numeric values using `LabelEncoder`.

- **Feature and Target Split**: 
  - The features (`X`) and labels (`y`) are extracted from the dataset. 
  - The features contain attributes such as buying price, maintenance cost, number of doors, etc.
  - The label (`y`) is the class representing the acceptability of the car (`class` column).

- **Data Splitting**: 
  - The dataset is split into training and testing sets using `train_test_split()`. 
  - 70% of the data is used for training, and 30% for testing.

- **Model Training**: 
  - A k-Nearest Neighbors classifier (`KNeighborsClassifier`) is initialized with `n_neighbors=5`.
  - The classifier is trained on the training set using `knn.fit()`.

- **Model Prediction**: 
  - Predictions are made on the test set using `knn.predict()`.

- **Evaluation**: 
  - The performance of the model is evaluated using the following metrics:
    - **Confusion Matrix**: Shows the number of correct and incorrect predictions.
    - **Accuracy**: The overall accuracy of the classifier.
    - **Precision**: The precision score for the classifier (weighted for multi-class classification).
    - **Recall**: The recall score (weighted for multi-class classification).
    - **F1-Score**: The harmonic mean of precision and recall (weighted for multi-class classification).

### Files
- **car.data**: The dataset containing car characteristics and their classifications.

### Instructions

1. **Install Required Libraries**:
   Make sure to install the required libraries if not already installed:
   ```bash
   pip install pandas scikit-learn
   ```

2. **Run the Notebook**: 
   Simply run each cell sequentially to load the data, preprocess it, train the model, and evaluate its performance.

3. **Customization**: 
   You can experiment with different hyperparameters (e.g., `n_neighbors` in the kNN classifier) and observe how they affect the performance metrics.

### Dependencies

- **Python 3.x**
- **pandas**
- **scikit-learn**
