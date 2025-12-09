# 🏦 Credit Risk Modelling & Loan Amount Predictor

A complete machine learning project that predicts **loan approval status** and **eligible loan amount** using applicant financial and demographic information.

---

## 📁 Project Structure

```txt
credit-risk-modelling-and-loan-amount-predictor/
│
├── artifacts/                    # Stored trained model files
│   └── model.pkl                 # Saved ML model
│
├── main.py                       # Main script for training the model
├── prediction_helper.py          # Script to load model & predict outputs
├── requirements.txt              # Project dependencies
├── LICENSE
└── README.md
```

---

## 🚀 Features

- End-to-end ML workflow (cleaning → preprocessing → model training → evaluation)
- Handles missing values, categorical encoding, scaling & normalization
- Trains multiple ML models (Logistic Regression, Random Forest, XGBoost)
- Predicts:
  - **Loan Status (Approved / Not Approved)**
  - **Loan Amount (if eligible)**
- Saves model to `artifacts/`
- Simple prediction script for quick testing

---

## 📊 Machine Learning Pipeline

### 🔍 Preprocessing
- Handle missing values  
- One-hot encoding  
- Standardization / normalization  
- Train/Test split  

### 📈 EDA
- Distribution plots  
- Correlation heatmap  
- Outlier detection  

### 🤖 Model Training
- Logistic Regression  
- Random Forest  
- XGBoost (optional)

### 🧪 Evaluation
- Accuracy  
- Confusion Matrix  
- Precision / Recall  
- ROC-AUC  

### 🔮 Prediction System
`prediction_helper.py` loads the model and predicts approval + loan amount.

---

## 🛠️ Technologies Used

### Core
- Python  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib, Seaborn  
- XGBoost  

### Optional
- Joblib/Pickle for saving models  
- Jupyter Notebook  

---

## 📦 Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/MothukuruPunith/credit-risk-modelling-and-loan-amount-predictor.git
cd credit-risk-modelling-and-loan-amount-predictor
```

### 2️⃣ Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3️⃣ Install Requirements
```bash
pip install -r requirements.txt
```

### 4️⃣ Train the Model
```bash
python main.py
```

Model will be saved to:
```
artifacts/model.pkl
```

### 5️⃣ Run Prediction Script
```bash
python prediction_helper.py
```

---

## 🔍 Example Output

```
Loan Approval Status: Approved
Predicted Loan Amount: ₹180,000
```

---

## 📄 requirements.txt (key packages)

```txt
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
joblib
```

---

## 👨‍💼 Author

**Punith Mothukuru**  
ML & GenAI Enthusiast  
SRM Institute of Science and Technology  

---

## 📬 Contributions

Feel free to open issues or contribute with pull requests.

---

## 📌 Note

This project is meant for learning and demonstration only.
