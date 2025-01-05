Fraud Detection Project 
A Python-based machine learning project to detect fraudulent credit card transactions using a public dataset from Kaggle.

Dataset 
The dataset used in this project is the Credit Card Fraud Detection Dataset from Kaggle.

It contains transactions made by European cardholders over two days, with 284,807 transactions and only 492 fraudulent cases (~0.17%).
The dataset is highly imbalanced and contains 30 numerical features (anonymized for confidentiality) and a target variable (Class), where:
0 = Non-Fraud
1 = Fraud
You can download the dataset from Kaggle here:
https://www.kaggle.com/mlg-ulb/creditcardfraud

Project Features 
Data Preprocessing:

Handled class imbalance using SMOTE (Synthetic Minority Oversampling Technique).
Normalized numerical features using StandardScaler.
Model Training:

Used Logistic Regression as the classification algorithm.
Split the dataset into training and testing sets (70/30).
Model Evaluation:

Evaluated the model using metrics like precision, recall, F1-score, and AUC-PR (Area Under the Precision-Recall Curve).
Visualized performance with a confusion matrix and precision-recall curve.
Results 
Achieved high precision and recall for detecting fraudulent transactions.
Precision-Recall AUC: e.g., 0.96 (update this with your results).
How to Run the Project 
Clone the repository:

Copy code
git clone https://github.com/PranayTyagi11/Fraud-Detection.git
cd Fraud-Detection
Install the required Python libraries:

Copy code
pip install -r requirements.txt
Download the dataset from Kaggle and save it in the data/ folder:

Copy code
data/creditcard.csv
Run the script:

Copy code
python fraud_detection.py
Check the results in the terminal and the generated output.txt file.

Requirements 
Python 3.x
Libraries:
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
Install all dependencies using:


Copy code
pip install -r requirements.txt

File Structure 
Fraud_Detection_Project/
├── data/
│   └── creditcard.csv      # Dataset (not included, download from Kaggle)
├── results/
│   └── output.txt          # Auto-generated results
├── fraud_detection.py      # Main script
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
Visualizations 
Class Distribution
The dataset is highly imbalanced, with far fewer fraud cases compared to non-fraud cases.
(Add your actual visualization)
