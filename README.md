# 📊 Credit Risk & Fraud Mitigation System

This repository features a high-precision machine learning pipeline designed to identify fraudulent credit card transactions in real-time. It specifically addresses the "needle in a haystack" challenge where fraud accounts for only **0.17%** of total volume.

## 🔗 Live Deployment
> **[🚀 Access the Interactive Fraud Simulator](https://gwtjbg8dxkfck8ee62v7mr.streamlit.app/)** > 

---

## 🚀 Executive Summary
* **Goal:** To capture 90% of fraud while remaining "invisible" to 99.97% of legitimate customers.
* **The Imbalance Challenge:** Initial data was 99.83% legitimate. To solve this, I utilized **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the training set from 339 fraud cases to 198,269 synthetic samples.
* **Architecture:** A **Random Forest** engine was selected as the superior solution for its ability to eliminate "false alarms" (96% Precision).

## 🔍 The "Signature of Fraud"
Using feature correlation analysis, the model identifies specific "red flags":
* **Strong Negative Predictors:** Features **V17, V14, and V12**—as these values decrease, the probability of fraud increases dramatically.
* **Strong Positive Predictors:** Features **V11 and V4**.

## 📈 Model Performance
The Random Forest model was benchmarked against three other architectures:

| Model | Recall (Fraud Caught) | Precision (Alert Accuracy) | False Positive Rate |
| :--- | :--- | :--- | :--- |
| **Random Forest** | **90.0%**  | **96.0%**  | **0.029%**  |
| Decision Tree | 82.0% | 75.0%  | 0.036%  |
| Logistic Regression | 91.0%  | 6.0%  | 3.615%  |
| Naive Bayes | 85.0%  | 5.0%  | 2.469%  |

## 🛡️ Strategic Risk Scoring
The system transitions from binary decisions to a **Dynamic Risk Score (0-100)** to optimize customer experience:
1. **Low Risk (<30):** Automated approval (Covers 99% of traffic).
2. **Medium Risk (30-85):** Trigger secondary friction (SMS/MFA).
3. **High Risk (>85):** Immediate hard-block for high-certainty theft.

---

## 🛠️ Tech Stack
* **Language:** Python
* **Data Science:** Pandas, NumPy, Scikit-learn (Random Forest)
* **Imbalance Handling:** Imbalanced-learn (SMOTE)
* **Visualization:** Matplotlib, Seaborn 
* **Deployment:** Streamlit / Flask

## 📁 File Descriptions
* `Credit_risk_final model.ipynb`: Full workflow including EDA, SMOTE resampling, and model benchmarking.
* `app.py`: Streamlit script for the interactive web application.
* `main.py`: Core logic for real-time model inference.

## ⚙️ How to Run Locally
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/obakeng1-code/Portfolio.git](https://github.com/obakeng1-code/Portfolio.git)
   cd Portfolio

