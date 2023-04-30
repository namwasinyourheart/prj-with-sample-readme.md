# Evaluating Risk for Loan Approvals

## Overview

This project aims to evaluate the risk of loan approvals based on a given dataset. The dataset contains information on past loan applications, such as loan amount, borrower income, loan purpose, and whether the loan was approved or not. The goal of this project is to build a machine learning model that can accurately predict whether a loan application will be approved or not based on the provided features. The project was implemented on Azure Databricks, utilizing PySpark for data processing and modeling.

## **Technologies Used**

- Azure Databricks
- PySpark
- Python
- Scikit-learn
- Pandas
- Matplotlib
- Seaborn

## **Dataset**

The dataset used in this project was oabtained from LendingClub, a peer-to-peer lending company that connects borrowers and investors. 

The dataset contains 2.26 million rows and 151 columns. The columns include borrower information such as loan amount, interest rate, employment status, credit score, … The target variable is the loan status, which indicates whether the loan was fully paid, charged off, or default.
The dataset can be found **[here](https://www.kaggle.com/wordsforthewise/lending-club)**.

The below image is shows a screenshot of the first few rows of the dataset. (or summary table showing the number of rows and columns, data types, and missing values.):

## **Methodology**

1. Data Cleaning and Preprocessing: The dataset is cleaned and preprocessed to remove missing values, outliers, and other inconsistencies. This step includes exploratory data analysis, feature engineering, and feature selection.
2. Data Exploration: Analyzed the distributions of variables, correlations between variables, and the relationship between variables and the target variable.
3. Model Selection and Training: Several machine learning models are trained on the preprocessed dataset, including Logistic Regression, Random Forest, and Gradient Boosted Trees. The models are evaluated based on various performance metrics, including accuracy, precision, recall, and F1-score.
4. Hyperparameter Tuning: The best-performing model is selected, and its hyperparameters are tuned using cross-validation.
5. Model Evaluation: The final model is evaluated on a test set, and its performance metrics are reported. The model is also tested on a new dataset to assess its generalization performance.

## **Results**

The final model **achieved an accuracy of 0.85 and an F1-score of 0.86 on the test set**. The **most important features** for predicting loan status were loan amount, debt-to-income ratio, and credit score.The model was able to predict whether a loan application would be approved or not with high accuracy and precision.

## Reproduction

To reproduce the results of this project, follow the steps below:

### **Environment Setup**

- Clone the repository: **`git clone https://github.com/username/repo.git`**
- Navigate to the project directory: **`cd repo`**
- Create a virtual environment: **`python3 -m venv venv`**
- Activate the virtual environment: **`source venv/bin/activate`**
- Install the required packages: **`pip install -r requirements.txt`**

### **Steps to Reproduce**

1. Download the dataset from [source link] and save it to **`data/`** folder.
2. Open Jupyter Notebook or any Python environment of your choice.
3. Navigate to the **`notebooks/`** folder and open **`loan_approval.ipynb`**.
4. Follow the instructions in the notebook to clean the data, explore the dataset, train the model and evaluate the results.
5. Alternatively, run the **`loan_approval.py`** script to reproduce the results.

Note: You may need to install additional Python libraries to run the notebook. These libraries are listed in the requirements.txt file in the GitHub repository.

## References

- LendingClub Loan Data from Kaggle: **[https://www.kaggle.com/wordsforthewise/lending-club](https://www.kaggle.com/wordsforthewise/lending-club)**
- CRISP-DM Methodology: **[https://www.sv-europe.com/crisp-dm-methodology/](https://www.sv-europe.com/crisp-dm-methodology/)**
- PySpark Documentation: **[https://spark.apache.org/docs/latest/api/python/index.html](https://spark.apache.org/docs/latest/api/python/index.html)**
- Scikit-learn Documentation: **[https://scikit-learn.org/stable/documentation.html](https://scikit-learn.org/stable/documentation.html)**
- Azure Databricks Documentation: **[https://docs.databricks.com/](https://docs.databricks.com/)**
