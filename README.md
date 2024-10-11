# ML_Project-Parkinson's_Disease_Detection 🧠🔬

This project explores the task of detecting Parkinson's disease using a **Support Vector Machine (SVM)** model. By analyzing various features derived from voice measurements, the model aims to accurately identify whether an individual is likely to have Parkinson's disease.

## Data
This directory contains the dataset (`Parkinsson disease.csv`) used for the project. The dataset includes the following features:

- **MDVP:Fo(Hz)**: Average vocal fundamental frequency.
- **MDVP:Fhi(Hz)**: Maximum vocal fundamental frequency.
- **MDVP:Flo(Hz)**: Minimum vocal fundamental frequency.
- **MDVP:Jitter(%)**: Several measures of variation in fundamental frequency.
- **MDVP:Jitter(Abs)**: Absolute jitter.
- **MDVP:RAP**: Relative amplitude perturbation.
- **MDVP:PPQ**: Five-point period perturbation quotient.
- **Jitter:DDP**: Average absolute differences between consecutive periods.
- **MDVP:Shimmer**: Measures of variation in amplitude.
- **Shimmer(dB)**: Amplitude perturbation quotient in dB.
- **Shimmer:APQ3**: Amplitude perturbation quotient based on 3-point periods.
- **Shimmer:APQ5**: Amplitude perturbation quotient based on 5-point periods.
- **Shimmer:DDA**: Average absolute differences between consecutive amplitude periods.
- **NHR**: Noise to harmonic ratio.
- **HNR**: Harmonic to noise ratio.
- **RPDE**: Recurrence period density entropy.
- **DFA**: Detrended fluctuation analysis.
- **spread1**, **spread2**, **D2**, **PPE**: Measures related to vocal irregularities.
- **Status**: Target variable (1 indicates Parkinson's, 0 indicates healthy).

> **Note:** You may need to adjust the dataset features based on your specific project requirements.

## Notebooks
This directory contains the Jupyter Notebook (`Parkinson's_Disease_Detection.ipynb`) that guides you through the entire process of data exploration, preprocessing, model training, evaluation, and visualization.

## Running the Project
The Jupyter Notebook (`Parkinson's_Disease_Detection.ipynb`) walks through the following steps:

### Data Loading and Exploration:
- Load the dataset and explore basic statistics.
- Visualize relationships between features and the target variable (`status`).

### Data Preprocessing:
- Handle missing values (if any).
- Scale numerical features like `MDVP:Fo(Hz)` and `MDVP:Shimmer`.
- Encode categorical variables (if applicable).

### Train-Test Split:
- The data is split into training and testing sets using `train_test_split` from the `sklearn` library, with a typical 80-20 or 70-30 ratio for training and testing, respectively.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Feature Engineering (Optional):
- Creates additional features (e.g., interactions between features).
- Analyzes correlations between features and the target variable.

### Model Training:
- Trains the model using **Support Vector Machine (SVM)**, potentially tuning hyperparameters for improved performance.

### Model Evaluation:
- Evaluates model performance using metrics like accuracy, precision, recall, and F1-score.

### Visualization of Results:
- Analyzes the confusion matrix to understand model performance on different categories.
- Visualizes feature importance to explore the impact of specific features on model predictions.

## Customization
Modify the Jupyter Notebook to:
- Experiment with different preprocessing techniques and feature engineering methods.
- Try other classification algorithms for comparison (e.g., **Random Forest**, **Logistic Regression**).
- Explore advanced techniques like deep learning models specifically designed for medical prediction.

## Resources
- Sklearn Documentation: [https://scikit-learn.org/stable/modules/svm.html](https://scikit-learn.org/stable/modules/svm.html)
- Parkinson’s Disease Dataset Source: [https://www.kaggle.com/datasets/debasisdotcom/parkinson-disease-detection](https://www.kaggle.com/datasets/debasisdotcom/parkinson-disease-detection)

## Further Contributions
Extend this project by:
- Incorporating additional medical data or biometric features for improved prediction.
- Implementing a real-time Parkinson’s detection system using a trained model and an API.
- Exploring explainability techniques to understand the reasoning behind the model's predictions.

By leveraging machine learning models, specifically **SVM** and voice data, we aim to develop a reliable method for detecting Parkinson’s disease. This project lays the foundation for further exploration into healthcare-based machine learning applications.
