# Customer Churn Prediction

## Overview
A machine learning project to predict customer churn using the Telco Customer Churn dataset.

## Tech Stack
- Python
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn (Logistic Regression, Random Forest, Gradient Boosting)
- Streamlit (Web App)

## Project Structure
- `churn_prediction.ipynb` — EDA + ML notebook
- `app.py` — Streamlit web application
- `telco_churn.csv` — Dataset

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Results
- Best Model: Gradient Boosting
- Accuracy: ~80%
- Features: Contract type, tenure, monthly charges are top predictors
