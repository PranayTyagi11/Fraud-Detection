# Fraud Detection Project 

A Python-based machine learning project to detect fraudulent credit card transactions using a public dataset from Kaggle.

## Dataset 

The dataset used in this project is the **Credit Card Fraud Detection Dataset** from Kaggle.

- It contains transactions made by European cardholders over two days, with 284,807 transactions and only 492 fraudulent cases (~0.17%).
- The dataset is highly imbalanced and contains 30 numerical features (anonymized for confidentiality) and a target variable (`Class`), where:
  - `0` = Non-Fraud
  - `1` = Fraud

You can download the dataset from Kaggle here:  
`https://www.kaggle.com/mlg-ulb/creditcardfraud`

## Project Features 

- **Data Preprocessing**:
  - Handled class imbalance using **SMOTE (Synthetic Minority Oversampling Technique)**.
  - Normalized numerical features using **StandardScaler**.

- **Model Training**:
  - Used **Logistic Regression** as the classification algorithm.
  - Split the dataset into training and testing sets (70/30).

- **Model Evaluation**:
  - Evaluated the model using metrics like **precision**, **recall**, **F1-score**, and **AUC-PR (Area Under the Precision-Recall Curve)**.
  - Visualized performance with a **confusion matrix** and **precision-recall curve**.

## Results 

- Achieved high precision and recall for detecting fraudulent transactions.
- **Precision-Recall AUC**: *e.g., 0.96* (update this with your results).

## How to Run the Project 

1. Clone the repository:
   ```bash
   git clone https://github.com/PranayTyagi11/Fraud-Detection.git
   cd Fraud-Detection

## Libraries

pandas, 
numpy, 
matplotlib, 
seaborn, 
scikit-learn, 
imbalanced-learn

## File structure

Fraud_Detection_Project/ data/creditcard.csv  # Downloaded from kaggle,   
Fraud_Detection_Project/results/ output.txt   # Auto-generated results,  
Fraud_Detection_Project/fraud_detection.py    # Main script 

