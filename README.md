# 📊 Credit Risk & Fraud Mitigation System

This repository features a high-precision machine learning pipeline designed to identify fraudulent credit card transactions in real-time. [cite_start]It specifically addresses the "needle in a haystack" challenge where fraud accounts for only **0.17%** of total volume[cite: 3, 22].

## 🔗 Live Deployment
> **[🚀 Access the Interactive Fraud Simulator](https://gwtjbg8dxkfck8ee62v7mr.streamlit.app/)** > 

---

## 🚀 Executive Summary
* [cite_start]**Goal:** To capture 90% of fraud while remaining "invisible" to 99.97% of legitimate customers[cite: 10, 12].
* [cite_start]**The Imbalance Challenge:** Initial data was 99.83% legitimate[cite: 22]. [cite_start]To solve this, I utilized **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the training set from 339 fraud cases to 198,269 synthetic samples[cite: 41, 48, 49].
* [cite_start]**Architecture:** A **Random Forest** engine was selected as the superior solution for its ability to eliminate "false alarms" (96% Precision)[cite: 8, 154].

## 🔍 The "Signature of Fraud"
[cite_start]Using feature correlation analysis, the model identifies specific "red flags"[cite: 7, 177]:
* [cite_start]**Strong Negative Predictors:** Features **V17, V14, and V12**—as these values decrease, the probability of fraud increases dramatically[cite: 7, 178].
* [cite_start]**Strong Positive Predictors:** Features **V11 and V4**[cite: 180].

## 📈 Model Performance
[cite_start]The Random Forest model was benchmarked against three other architectures[cite: 115]:

| Model | Recall (Fraud Caught) | Precision (Alert Accuracy) | False Positive Rate |
| :--- | :--- | :--- | :--- |
| **Random Forest** | [cite_start]**90.0%** [cite: 235] | [cite_start]**96.0%** [cite: 236] | [cite_start]**0.029%** [cite: 238] |
| Decision Tree | [cite_start]82.0% [cite: 240] | [cite_start]75.0% [cite: 241] | [cite_start]0.036% [cite: 243] |
| Logistic Regression | [cite_start]91.0% [cite: 250] | [cite_start]6.0% [cite: 251] | [cite_start]3.615% [cite: 253] |
| Naive Bayes | [cite_start]85.0% [cite: 245] | [cite_start]5.0% [cite: 246] | [cite_start]2.469% [cite: 248] |

## 🛡️ Strategic Risk Scoring
[cite_start]The system transitions from binary decisions to a **Dynamic Risk Score (0-100)** to optimize customer experience[cite: 13, 284]:
1. [cite_start]**Low Risk (<30):** Automated approval (Covers 99% of traffic)[cite: 14, 341].
2. [cite_start]**Medium Risk (30-85):** Trigger secondary friction (SMS/MFA)[cite: 15, 344].
3. [cite_start]**High Risk (>85):** Immediate hard-block for high-certainty theft[cite: 16, 345].

---

## 🛠️ Tech Stack
* **Language:** Python
* **Data Science:** Pandas, NumPy, Scikit-learn (Random Forest)
* [cite_start]**Imbalance Handling:** Imbalanced-learn (SMOTE) [cite: 5, 33]
* [cite_start]**Visualization:** Matplotlib, Seaborn [cite: 100]
* **Deployment:** Streamlit / Flask

## 📁 File Descriptions
* [cite_start]`Credit_risk_final model.ipynb`: Full workflow including EDA, SMOTE resampling, and model benchmarking[cite: 18, 54].
* `app.py`: Streamlit script for the interactive web application.
* `main.py`: Core logic for real-time model inference.

## ⚙️ How to Run Locally
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/obakeng1-code/Portfolio.git](https://github.com/obakeng1-code/Portfolio.git)
   cd Portfolio
