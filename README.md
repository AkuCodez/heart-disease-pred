# 💓 Heart Disease Prediction System

A Machine Learning-powered web app that predicts whether a patient is at risk of heart disease, based on clinical and diagnostic attributes. Built using **Python**, **Scikit-learn**, and deployed with **Streamlit Cloud** for real-time access.

![Streamlit App Screenshot](![image](https://github.com/user-attachments/assets/3ada0c80-e487-4467-bc5e-7416635123a6)

---

## 🚀 Live Demo

👉 [Click here to try the live app](https://your-username-your-app-name.streamlit.app)

---

## 🧠 Features

- Predict heart disease based on 11 medical inputs
- Clean, responsive Streamlit UI
- Tooltips for each feature to help users understand inputs
- Real-time prediction with model confidence score
- Option to compare prediction with actual diagnosis
- Deployed free with Streamlit Cloud

---

## 🗂️ Tech Stack

| Tool           | Purpose                                 |
|----------------|------------------------------------------|
| `Python`       | Programming language                     |
| `Scikit-learn` | Machine learning model (Random Forest)   |
| `Pandas`       | Data manipulation                        |
| `Joblib`       | Model serialization                      |
| `Streamlit`    | Web app framework                        |

---

## 📁 Project Structure
heart-disease-app/
├── app.py # Streamlit web app
├── heart_disease_model.pkl # Trained Random Forest model
├── scaler.pkl # StandardScaler for input features
├── requirements.txt # Project dependencies
└── README.md # Project overview

## **Installation**
# 1. Clone the repository
git clone https://github.com/your-username/heart-disease-predictor.git
cd heart-disease-predictor

# 2. (Optional) Create a virtual environment
python -m venv venv
venv\Scripts\activate   # For Windows
# source venv/bin/activate  # For macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app.py

Model: RandomForestClassifier (from scikit-learn)
-----------------------------------------------
- n_estimators = 100
- random_state = 42
- Trained on scaled feature set using StandardScaler

Performance:
------------
- Accuracy: ~95% on test set (80/20 split)
- Evaluation Metrics:
    • Precision: ~0.95
    • Recall: ~0.96
    • F1-score: ~0.95
- Target Variable:
    0 = No Heart Disease
    1 = Has Heart Disease
