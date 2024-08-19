# Natural Disaster Prediction

## Project Overview
This project aims to predict various types of natural disasters (Wildfire, Cyclone, Earthquake, Flood) using machine learning techniques. The analysis includes exploratory data analysis, data preprocessing, feature selection, model development, and evaluation.

## Dataset
The dataset contains information about different types of natural disasters across various continents. It includes features such as disaster type, location, and frequency.

## Methodology

### Step 1: Exploratory Data Analysis (EDA)
- **Bar Plot for Frequency of Disaster Types by Continent**: 
  - Continents where disasters occurred more, in descending order:
    - Asia: 6490
    - Americas: 3971
    - Africa: 2946
    - Europe: 1997
    - Oceania: 722

- **Horizontal Bar Chart for Distribution of Disaster Types**
- **Correlation Analysis**: Heatmap to visualize relationships between features.
- **Time Series Analysis**: Analyzed the top 5 disasters over time.

### Step 2: Data Preprocessing
- **Checking for Missing Values**: Identified and handled missing data.
- **Handling Missing Values**: 
  - Imputed missing numerical values with the mean.
  - Imputed missing categorical values with the mode.
- **Encoding Categorical Variables**: Converted categorical variables into numerical format.

### Step 3: Feature Selection
- Selected features based on Mutual Information and domain knowledge.

### Step 4: Model Development
- **Random Forest Classifier**: 
  - Accuracy: 83.17%
- **Support Vector Machine (SVM)**: 
  - Accuracy: 37.17%
- **K-Nearest Neighbors (KNN)**: 
  - Accuracy: 68.94%
- **Naive Bayes**: 
  - Accuracy: 14.29%

### Step 5: Model Evaluation
#### Random Forest Classifier Evaluation Metrics:
- F1 Score: 0.82
- Accuracy: 83.17%
- Recall (Sensitivity): 83.17%
- Precision: 81.19%

#### Support Vector Machine (SVM) Evaluation Metrics:
- F1 Score: 0.22
- Accuracy: 37.17%
- Recall (Sensitivity): 37.17%
- Precision: 15.79%

#### K-Nearest Neighbor (K-NN) Evaluation Metrics:
- F1 Score: 0.67
- Accuracy: 68.94%
- Recall (Sensitivity): 68.94%
- Precision: 66.56%

#### Naive Bayes Evaluation Metrics:
- F1 Score: 0.16
- Accuracy: 14.29%
- Recall (Sensitivity): 14.29%
- Precision: 36.79%

### Step 6: Tuning the Models for Better Results
- Checked data balance and applied Random Oversampler technique to balance the dataset.

#### After Balancing Data:
- **Random Forest Classifier**:
  - Accuracy: 95.38%
  - F1 Score: 0.95
  - Recall: 95.38%
  - Precision: 95.52%

- **Support Vector Machine (SVM)**:
  - Accuracy: 65.01%
  - F1 Score: 0.63
  - Recall: 65.01%
  - Precision: 68.31%

- **K-Nearest Neighbor (K-NN)**:
  - Accuracy: 93.13%
  - F1 Score: 0.93
  - Recall: 93.13%
  - Precision: 93.16%

- **Naive Bayes**:
  - Accuracy: 65.32%
  - F1 Score: 0.63
  - Recall: 65.32%
  - Precision: 71.67%

### Comparing the Performance After Tuning
- **Ensemble Techniques**:
  - Hard Voting Ensemble: 
    - Accuracy: 92.36%
    - F1 Score: 0.92
  - Soft Voting Ensemble: 
    - Accuracy: 93.91%
    - F1 Score: 0.93

### Final Model Evaluation
- **Random Forest Classifier after Hyperparameter Tuning**:
  - Accuracy: 95.38%
  - F1 Score: 0.95
  - Recall: 95.38%
  - Precision: 95.52%

### Step 7: Saving the Model
- Loaded new unseen data, preprocessed it, and applied the saved model.
- **Classification Report on New Data**:
          precision    recall  f1-score   support
       0       1.00      1.00      1.00      5272
       1       0.91      0.90      0.91      5272
       2       1.00      1.00      1.00      5272
       3       1.00      1.00      1.00      5272
       4       1.00      1.00      1.00      5272
       5       0.99      0.75      0.85      5272
       6       1.00      1.00      1.00      5272
       7       1.00      1.00      1.00      5272
       8       0.95      0.98      0.96      5272
       9       0.84      0.87      0.85      5272
      10       0.93      1.00      0.97      5272
      11       1.00      1.00      1.00      5272
      12       0.91      0.98      0.94      5272
      13       0.93      0.95      0.94      5272
accuracy                           0.96     73808

## Technologies Used
- Python
- Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

## How to Use
1. Run the Jupyter notebook or Python script

## Future Work
- Explore more advanced feature engineering techniques
- Investigate deep learning approaches for disaster prediction
- Develop a real-time prediction system


This project demonstrates the application of machine learning techniques in predicting natural disasters, providing valuable insights for disaster preparedness and management.


