# Fraud Detection Using Autoencoders

## 📌 Project Overview

Fraudulent credit card transactions are rare compared with normal transactions, making traditional supervised fraud detection challenging when reliable fraud labels are unavailable during training.

This project implements an **unsupervised fraud/anomaly detection system using an Autoencoder**. The model learns patterns from **normal transactions only** and identifies unusual transactions using **reconstruction error**.

The project also compares the Autoencoder with **Isolation Forest** and **One-Class SVM** and provides an interactive **Streamlit dashboard**.

> **Important:** Fraud labels are not used for Autoencoder training. They are used only for held-out evaluation.

---

## 🎯 Project Objectives

- Train an Autoencoder using normal transactions only.
- Detect unusual transactions using reconstruction error.
- Evaluate the model using fraud labels only during testing.
- Compare Autoencoder, Isolation Forest and One-Class SVM.
- Build a Streamlit dashboard for detection, visualization and evaluation.

---

## 📊 Dataset

The project uses the **Credit Card Fraud Detection Dataset** from Kaggle.

:contentReference[oaicite:0]{index=0}

| Property | Value |
|---|---:|
| Total Transactions | 284,807 |
| Features | 30 |
| Normal Transactions | 284,315 |
| Fraud Transactions | 492 |
| Missing Values | 0 |
| Fraud Percentage | 0.1727% |

The dataset is highly imbalanced, with fraud representing a very small percentage of transactions.

---

## 🧠 Autoencoder

The Autoencoder learns the patterns of normal transactions and reconstructs the input.

### Architecture

```text
30 → 16 → 8 → 16 → 30

🧪 Training Methodology

An 80/20 stratified train-test split is used.

Training
Total training transactions: 227,845
Normal transactions used for training: 227,451

Only normal transactions from the training set are used to train the Autoencoder.

Held-Out Testing
Total test transactions: 56,962
Normal transactions: 56,864
Fraud transactions: 98

Fraud labels are used only after predictions are generated.

🚨 Anomaly Detection

The Autoencoder calculates a reconstruction error for each transaction.

Reconstruction Error > Threshold
              ↓
           Anomaly

The threshold is based on the 95th percentile of normal training errors.

Final threshold: 1.0739

📈 Autoencoder Results
Metric	Result
Accuracy	94.75%
Precision	2.78%
Recall	        86.73%
F1 Score	5.38%
Fraud Caught	85 / 98
Confusion Matrix
	       Predicted Normal	  Predicted Anomaly
Actual Normal	53,886	          2,978
Actual Fraud	13	          85

The model detected 85 out of 98 fraud cases in the held-out test set.

🆚 Model Comparison
Model	                  Accuracy	    Precision	    Recall	          F1 Score     Fraud Caught
Autoencoder	           94.75%	      2.78%	    86.73%	           5.38%	   85/98
Isolation Forest	   94.76%	      2.81%	    87.76%	           5.45%	   86/98
One-Class SVM	           93.26%	      2.25%	    89.80%	           4.38%	   88/98
Best F1: Isolation Forest – 5.45%
Highest Recall: One-Class SVM – 89.80%
Primary Model: Autoencoder
🌐 Streamlit Dashboard

The dashboard provides:

Dataset overview and transaction preview
Anomaly detection
Reconstruction error visualization
Flagged transaction details
Evaluation metrics and confusion matrix
Model comparison
CSV download

For flagged transactions, the dashboard explains that the reconstruction error exceeds the threshold.

🛠️ Technologies Used
Technology	Purpose
Python	Main programming language
Pandas, NumPy	Data processing and numerical operations
TensorFlow / Keras	Autoencoder and deep learning
Scikit-learn	Scaling, evaluation and ML models
Isolation Forest	Anomaly detection comparison
One-Class SVM	Anomaly detection comparison
Streamlit	Interactive dashboard
Altair	Data visualization
Joblib	Saving/loading scaler
Git & GitHub	Version control and repository
VS Code	Development environment

📁 Project Structure
Fraud-Detection-Autoencoder/
│
├── data/
│   └── creditcard.csv
│
├── models/
│   ├── autoencoder.keras
│   ├── scaler.pkl
│   ├── threshold.txt
│   ├── training_history.csv
│   ├── evaluation_results.csv
│   ├── model_comparison.csv
│   └── test_indices.csv
│
├── src/
│   ├── preprocessing.py
│   ├── autoencoder.py
│   ├── anomaly_detection.py
│   ├── evaluate.py
│   └── compare_models.py
│
├── train.py
├── app.py
├── README.md
├── TASK4_WRITEUP.md
├── requirements.txt
└── .gitignore

⚙️ Installation

Install the required packages:

pip install -r requirements.txt

Download the Kaggle dataset and place it at:

data/creditcard.csv

The dataset is excluded from GitHub because of its large size.

▶️ How to Run
Train the model
python train.py
Evaluate the model
python src/evaluate.py
Compare models
python src/compare_models.py
Run the dashboard
python -m streamlit run app.py

🔐 Important Fraud-Label Rule

Fraud labels are not used during Autoencoder training.

They are used only on the held-out test set to measure:

Precision
Recall
F1 Score
Fraud cases detected
Confusion matrix

🏆 Conclusion

This project demonstrates an unsupervised Autoencoder-based approach to fraud detection without using fraud labels during training.

The final Autoencoder achieved:

94.75% Accuracy
86.73% Recall
5.38% F1 Score
85/98 Fraud Cases Detected

The project combines Python, Machine Learning, Deep Learning, Data Processing, Model Evaluation, Visualization, Streamlit and GitHub into a complete fraud detection application.