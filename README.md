🏦 Credit Risk Modelling & Loan Amount Predictor

This project focuses on predicting loan approval amounts and assessing credit risk using machine learning techniques. It includes end-to-end steps from data preprocessing, EDA, feature engineering, model training, and evaluation.
📁 Project Structure
credit-risk-modelling-and-loan-amount-predictor/
│
├── data/                     # Dataset used for training & evaluation
│   └── loan_data.csv         
│
│
├── src/                      # Core Python scripts
│   ├── data_preprocessing.py # Handling missing values, encoding, scaling
│   ├── model.py              # ML model building & evaluation
│   └── utils.py              # Helper functions
│
├── models/                   # Saved trained models
│   └── loan_model.pkl
│
├── requirements.txt          # Python dependencies
└── README.md
Clean and preprocess raw loan applicant data

Perform detailed exploratory data analysis

Handle missing values, outliers & categorical encoding

Feature engineering to improve model accuracy

Train multiple ML models (Random Forest, XGBoost, etc.)

Predict:

Whether a loan should be approved

Approximate loan amount for approved applicants

Model evaluation with accuracy, precision, recall, ROC-AUC

Export the trained model for deployment

🧠 Machine Learning Techniques Used

Classification Models

Logistic Regression

Random Forest Classifier

XGBoost Classifier

Regression Models

Linear Regression

Random Forest Regressor

Preprocessing

One-hot encoding

Standardization / normalization

Train/Test split

🛠️ Technologies Used
Core

Python

NumPy & Pandas (data manipulation)

Matplotlib & Seaborn (EDA & visualization)

Scikit-learn (model training)

XGBoost (advanced modelling)

Optional

Jupyter Notebook for step-by-step modelling

Joblib/Pickle for model persistence

📦 Installation & Setup
1. Clone the Repository
2. Create Virtual Environment
   python -m venv venv
   source venv/bin/activate  # Mac/Linux
   venv\Scripts\activate     # Windows

3. Install Requirements
   pip install -r requirements.txt
▶️ Running the Project
python src/data_preprocessing.py
python src/model.py

👨‍💼 Author

Name: Punith
Domain: Machine Learning & Credit Risk Modelling
Institution: SRM Institute of Science and Technology

📬 Feedback & Contributions
Pull requests and suggestions are welcome.
If you find issues, feel free to raise them!

📌 Note

This project is intended for educational & analytical purposes.
Dataset values may be synthetic or anonymized.
Not intended for real-world financial decisions.
