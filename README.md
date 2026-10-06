# 💰 Loan Approval Prediction

## 📌 Overview

This project uses machine learning to predict whether a loan application will be approved based on applicant and loan-related information.

The project includes data preprocessing, exploratory data analysis, feature engineering, feature scaling, model training, and model evaluation.

## 🎯 Problem Statement

The objective of this project is to build classification models that can predict the `Loan_Approved` outcome using information about the applicant and loan application.

## 📊 Dataset

The project uses:

`loan_approval_data.csv`

The dataset contains 1000 rows and 20 columns before preprocessing.

### 🎯 Target Variable

`Loan_Approved`

### 🧾 Features

- Applicant_ID
- Applicant_Income
- Coapplicant_Income
- Employment_Status
- Age
- Marital_Status
- Dependents
- Credit_Score
- Existing_Loans
- DTI_Ratio
- Savings
- Collateral_Value
- Loan_Amount
- Loan_Term
- Loan_Purpose
- Property_Area
- Education_Level
- Gender
- Employer_Category

`Applicant_ID` is removed before model training.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## 🔄 Project Workflow

1. Load the dataset
2. Inspect the dataset
3. Handle missing values
4. Perform exploratory data analysis
5. Encode categorical variables
6. Remove unnecessary features
7. Analyze feature correlations
8. Split data into training and testing sets
9. Scale the features
10. Train machine learning models
11. Evaluate the models
12. Perform feature engineering
13. Train and evaluate the models again

## 🧹 Data Preprocessing

### 🧩 Missing Values

Numerical missing values are handled using the mean:

`SimpleImputer(strategy="mean")`

Categorical missing values are handled using the most frequent value:

`SimpleImputer(strategy="most_frequent")`

### 🗑️ Removing Applicant ID

The `Applicant_ID` column is removed because it is not used as a predictive feature.

### 🔤 Label Encoding

`LabelEncoder` is used for:

- `Education_Level`
- `Loan_Approved`

### 🔢 One-Hot Encoding

`OneHotEncoder` is used for:

- Employment_Status
- Marital_Status
- Loan_Purpose
- Property_Area
- Gender
- Employer_Category

The encoder uses `drop="first"` and `handle_unknown="ignore"`.

## 🔍 Exploratory Data Analysis

The project performs exploratory data analysis using Matplotlib and Seaborn.

The analysis includes:

- Dataset inspection
- Descriptive statistics
- Missing-value analysis
- Applicant income distribution
- Applicant income based on loan approval
- Correlation analysis
- Correlation heatmap

## ⚙️ Feature Engineering

Two additional features are created:

- `DTI_Ratio_sq`
- `Credit_Score_sq`

They are calculated as:

`DTI_Ratio_sq = DTI_Ratio²`

`Credit_Score_sq = Credit_Score²`

The original `DTI_Ratio` and `Credit_Score` features are then removed from the feature set for the second model training stage.

The notebook also contains a commented-out logarithmic transformation for applicant income:

`np.log1p(Applicant_Income)`

## ✂️ Train-Test Split

The dataset is divided into training and testing sets using an 80:20 split.

The split uses:

`random_state=42`

## 📏 Feature Scaling

The project uses `StandardScaler` for feature scaling.

The scaler is fitted on the training data and then used to transform the test data.

## 🤖 Machine Learning Models

### 📈 Logistic Regression

`LogisticRegression()`

### 👥 K-Nearest Neighbors

`KNeighborsClassifier(n_neighbors=5)`

The KNN model uses 5 neighbors.

### 🧠 Gaussian Naive Bayes

`GaussianNB()`

## 📋 Model Evaluation

The models are evaluated using:

- Precision
- Recall
- F1 Score
- Accuracy
- Confusion Matrix

## 📊 Results Before Feature Engineering

| 🤖 Model | 🎯 Precision | 🔄 Recall | 📊 F1 Score | ✅ Accuracy |
|:---|---:|---:|---:|---:|
| Logistic Regression | 0.7833 | 0.7705 | 0.7769 | 86.5% |
| K-Nearest Neighbors (KNN) | 0.6275 | 0.5246 | 0.5714 | 76.0% |
| Gaussian Naive Bayes | 0.8036 | 0.7377 | 0.7692 | 86.5% |

### 🔲 Confusion Matrices

**Logistic Regression**

```text
[[126  13]
 [ 14  47]]
```

**KNN**

```text
[[120  19]
 [ 29  32]]
```

**Naive Bayes**

```text
[[128  11]
 [ 16  45]]
```

## 🚀 Results After Feature Engineering

| 🤖 Model | 🎯 Precision | 🔄 Recall | 📊 F1 Score | ✅ Accuracy |
|:---|---:|---:|---:|---:|
| Logistic Regression | 0.7846 | 0.8361 | 0.8095 | 88.0% |
| K-Nearest Neighbors (KNN) | 0.6731 | 0.5738 | 0.6195 | 78.5% |
| Gaussian Naive Bayes | 0.8113 | 0.7049 | 0.7544 | 86.0% |

### 🔎 Model Comparison

| 🤖 Model | 📌 Before FE Accuracy | 🚀 After FE Accuracy | 📈 Accuracy Change |
|:---|---:|---:|---:|
| Logistic Regression | 86.5% | 88.0% | +1.5% |
| K-Nearest Neighbors (KNN) | 76.0% | 78.5% | +2.5% |
| Gaussian Naive Bayes | 86.5% | 86.0% | -0.5% |

> **FE = Feature Engineering**

### 🔲 Confusion Matrices

**Logistic Regression**

```text
[[125  14]
 [ 10  51]]
```

**KNN**

```text
[[122  17]
 [ 26  35]]
```

**Naive Bayes**

```text
[[129  10]
 [ 18  43]]
```

## 📦 Installation

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

## ▶️ How to Run

1. Download or clone the project.
2. Place `loan_approval_data.csv` in the project directory.
3. Install the required Python libraries.
4. Start Jupyter Notebook:

```bash
jupyter notebook
```

5. Open the project `.ipynb` file.
6. Run the notebook cells in order.

## 📁 Project Structure

```text
Loan-Approval-Prediction/
│
├── loan_approval_data.csv
├── <notebook>.ipynb
└── README.md
```

## ✅ Conclusion

This project demonstrates a machine learning workflow for loan approval prediction, including data preprocessing, exploratory data analysis, categorical encoding, feature scaling, feature engineering, model training, and model evaluation.

The implemented models are Logistic Regression, K-Nearest Neighbors, and Gaussian Naive Bayes. Their performance is evaluated before and after feature engineering using precision, recall, F1 score, accuracy, and confusion matrices.
