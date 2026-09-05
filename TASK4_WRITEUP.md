# Task 4 Write-Up
# Fraud Detection Using Autoencoders

## 1. Introduction

Fraud detection is a challenging machine learning problem because fraudulent transactions are rare compared with legitimate transactions.

The objective of this project is to build an anomaly detection system that can identify unusual credit card transactions without depending on labelled fraud examples during model training.

An Autoencoder is used as the primary unsupervised model. The model learns to reconstruct normal transactions. Transactions that produce unusually high reconstruction errors are considered anomalies.

The project also compares the Autoencoder with Isolation Forest and One-Class SVM and provides a Streamlit dashboard for interactive fraud and anomaly analysis.

---

## 2. Problem Statement

The Task 4 project requires an anomaly detection system that learns what normal transactions look like and flags unusual transactions without using fraud labels during training.

The key challenge is the severe class imbalance in the dataset. There are far more legitimate transactions than fraudulent transactions.

Therefore, instead of training a conventional supervised classifier using fraud labels, this project uses an unsupervised Autoencoder to learn normal transaction behaviour.

The system uses reconstruction error to determine whether a transaction is normal or anomalous.

---

## 3. Project Objectives

The main objectives of this project are:

- Detect unusual credit card transactions without using fraud labels during training.
- Train an Autoencoder using normal transactions only.
- Calculate reconstruction error for each transaction.
- Select an anomaly threshold using normal training reconstruction errors.
- Evaluate the trained model on an unseen held-out test set.
- Measure Precision, Recall and F1 Score.
- Count the number of known fraud cases detected.
- Compare the Autoencoder with Isolation Forest and One-Class SVM.
- Build an interactive Streamlit dashboard.
- Display flagged transactions and explain why they were flagged.

---

## 4. Dataset

The project uses the Kaggle Credit Card Fraud Detection dataset.

Dataset characteristics:

- Total transactions: 284,807
- Total columns: 31
- Transaction features used by the model: 30
- Normal transactions: 284,315
- Fraud transactions: 492
- Missing values: 0

The dataset is highly imbalanced because fraudulent transactions represent only a very small portion of all transactions.

The `Class` column represents the known fraud label.

However, the `Class` column is not used as an input feature for the Autoencoder.

The fraud labels are used only for final evaluation on the held-out test set.

Dataset source:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

## 5. Train/Test Methodology

A formal 80/20 train-test split is used in the project.

The dataset is divided into:

- 80% training data
- 20% held-out test data

A stratified split is used so that the rare fraud class is represented in both portions.

### Training Set

The training set contains:

- Total training transactions: 227,845
- Normal training transactions: 227,451
- Fraud transactions: 394

The Autoencoder is trained using only the 227,451 normal training transactions.

The fraud labels from the training portion are not used by the Autoencoder for learning.

### Held-Out Test Set

The held-out test set contains:

- Total test transactions: 56,962
- Normal transactions: 56,864
- Fraud transactions: 98

The held-out test set is not used for Autoencoder training or threshold selection.

It is used only for final evaluation.

This allows the model to be tested on transactions that were not used during model training.

---

## 6. Data Preprocessing

The `Class` column is separated from the transaction features.

The model uses the remaining 30 transaction features.

The preprocessing workflow is:

1. Load the credit card transaction dataset.
2. Separate the transaction features from the `Class` label.
3. Create an 80/20 stratified train-test split.
4. Select normal transactions from the training set.
5. Fit `StandardScaler` using only normal training transactions.
6. Transform the normal training data using the fitted scaler.
7. Transform the held-out test data using the same scaler.
8. Train the Autoencoder using only normal training transactions.

Scaling is important because the transaction features have different numerical ranges.

Fitting the scaler only on the normal training data prevents the held-out test data from influencing the preprocessing parameters.

---

## 7. Autoencoder Architecture

The primary anomaly detection model is a feed-forward Autoencoder implemented using TensorFlow/Keras.

The architecture is:

```text
Input Layer
30 features
     ↓
Dense Layer
16 neurons
     ↓
Dense Layer
8 neurons
     ↓
Dense Layer
16 neurons
     ↓
Output Layer
30 features