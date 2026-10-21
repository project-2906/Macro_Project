# Macro_Project
# 🏦 Bank Customer Churn Prediction System

## 📌 Project Overview
Customer churn is a critical challenge in the banking industry. Acquiring a new customer is significantly more expensive than retaining an existing one. This project builds an end-to-end Machine Learning pipeline and a web-based dashboard to predict whether a bank customer is likely to leave (churn) based on their historical data.

By identifying at-risk customers proactively, banks can deploy targeted retention strategies, reduce revenue loss, and improve customer lifetime value.

## 🚀 Live Demo (Local Setup)
The project includes an interactive Streamlit dashboard that allows users to make single predictions or process batch predictions via CSV upload.

## 📊 Dataset
The project uses the **Bank Customer Churn Modelling** dataset (10,000 rows, 14 columns). 
*   **Features:** Credit Score, Geography, Gender, Age, Tenure, Balance, Number of Products, Has Credit Card, Is Active Member, Estimated Salary.
*   **Target:** `Exited` (1 = Customer left, 0 = Customer stayed).
*   **Challenge:** The dataset suffers from class imbalance (~20% churn rate), which was addressed using SMOTE.

## 🛠️ Tech Stack & Libraries
*   **Language:** Python 3.10+
*   **Data Processing:** Pandas, NumPy
*   **Machine Learning:** Scikit-Learn, XGBoost, Imbalanced-Learn (SMOTE)
*   **Visualization:** Matplotlib, Seaborn, Plotly
*   **Frontend Web App:** Streamlit
*   **Model Saving:** Joblib

## 🧠 Model Performance & Selection
Four classification algorithms were evaluated. The **XGBoost Classifier** was selected as the final model due to its superior ability to handle complex, non-linear patterns and imbalanced data.

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 71.8% | 38.8% | 67.3% | 49.2% |
| Decision Tree | 80.3% | 51.1% | 73.7% | 60.3% |
| Random Forest | 80.9% | 52.2% | 73.2% | 61.0% |
| **XGBoost (Winner)** | **84.55%** | **61.6%** | **63.6%** | **62.6%** |

*Note: Hyperparameter tuning (GridSearchCV) and Feature Engineering were also tested but resulted in lower accuracy due to overfitting and noise.*

## 📁 Repository Structure
```text
├── Customer_Churn_Prediction.ipynb  # Main Jupyter Notebook (EDA, Cleaning, Modeling)
├── app.py                           # Streamlit Frontend Code
├── churn_xgb_model.pkl              # Trained XGBoost Model (Pickle file)
├── scaler.pkl                       # Fitted StandardScaler (Pickle file)
└── README.md                        # Project Documentation
