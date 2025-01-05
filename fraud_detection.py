# Author: Pranay Tyagi
# Description: A Python script to detect fraudulent transactions using machine learning.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
from imblearn.over_sampling import SMOTE

data = pd.read_csv('data/creditcard.csv')

print("Dataset Information:")
print(data.info())

print("\nFirst 5 rows of the dataset:")
print(data.head())

missing_values = data.isnull().sum()
print("\nMissing Values in Dataset:")
print(missing_values)

class_counts = data['Class'].value_counts()
print("\nClass Distribution:")
print(class_counts)

class_counts.plot(kind='bar', color=['blue', 'orange'])
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.xticks([0, 1], ['Non-Fraud', 'Fraud'], rotation=0)
plt.show()

correlation_matrix = data.corr()
print("\nCorrelation with Target (Class):")
print(correlation_matrix['Class'].sort_values(ascending=False))

X = data.drop('Class', axis=1)
y = data['Class']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

print("\nData successfully preprocessed and split!")
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE Resampling:")
print(f"Resampled training set size: {X_resampled.shape[0]} samples")
print(f"Fraud cases in resampled data: {sum(y_resampled == 1)}")

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_resampled, y_resampled)

print("\n--- Debug: Starting Predictions ---")
y_pred = model.predict(X_test)
print("Predictions Complete")

print("\n--- Model Evaluation ---")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Visualize the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Plot Precision-Recall Curve
y_scores = model.predict_proba(X_test)[:, 1]
precision, recall, _ = precision_recall_curve(y_test, y_scores)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.', label='Logistic Regression')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()

# Calculate AUC for Precision-Recall Curve
pr_auc = auc(recall, precision)
print(f"Precision-Recall AUC: {pr_auc:.2f}")

# Save outputs to a file
with open("output.txt", "w") as f:
    f.write("\n--- Model Evaluation ---\n")
    f.write("\nClassification Report:\n")
    f.write(classification_report(y_test, y_pred))
    f.write("\nConfusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, y_pred)))
    f.write(f"\nPrecision-Recall AUC: {pr_auc:.2f}\n")

# Debug: Confirm script completed
print("\n--- Script Completed Successfully ---")
print("Results saved to output.txt")
