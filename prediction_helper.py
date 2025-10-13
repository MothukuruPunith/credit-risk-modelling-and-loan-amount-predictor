import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Path to the saved model and its components
MODEL_PATH = 'artifacts/model_data.joblib'

# Load the model and its components
model_data = joblib.load(MODEL_PATH)
model = model_data['model']
scaler = model_data['scaler']
features = model_data['features']
cols_to_scale = model_data['cols_to_scale']


def prepare_input(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
                    delinquency_ratio, credit_utilization_ratio, num_open_accounts, residence_type,
                    loan_purpose, loan_type , years_at_current_address,employment_status):
    # Create a dictionary with input values and dummy values for missing features
    input_data = {
        'age': age,
        'loan_tenure_months': loan_tenure_months,
        'number_of_open_accounts': num_open_accounts if num_open_accounts>0 else 0,
        'credit_utilization_ratio': credit_utilization_ratio,
        'loan_to_income': loan_amount / income if income > 0 else 0,
        'delinquency_ratio': delinquency_ratio,
        'avg_dpd_per_delinquency': avg_dpd_per_delinquency,
        'residence_type_Owned': 1 if residence_type == 'Owned' else 0,
        'residence_type_Rented': 1 if residence_type == 'Rented' else 0,
        'loan_purpose_Education': 1 if loan_purpose == 'Education' else 0,
        'loan_purpose_Home': 1 if loan_purpose == 'Home' else 0,
        'loan_purpose_Personal': 1 if loan_purpose == 'Personal' else 0,
        'loan_type_Unsecured': 1 if loan_type == 'Unsecured' else 0,
        'years_at_current_address' : years_at_current_address,
        'employment_status' : 1 if employment_status == 'Salaried' else 0,
        'employment_status_Self-Employed' : 1 if employment_status == 'Self-Employed' else 0,
        # additional dummy fields just for scaling purpose
        'number_of_dependants': 1,  # Dummy value
         # Dummy value
        'zipcode': 1,  # Dummy value
        'sanction_amount': 1,  # Dummy value
        'processing_fee': 1,  # Dummy value
        'gst': 1,  # Dummy value
        'net_disbursement': 1,  # Computed dummy value
        'principal_outstanding': 1,  # Dummy value
        'bank_balance_at_application': 1,  # Dummy value
        'number_of_closed_accounts': 1,  # Dummy value
        'enquiry_count': 1  # Dummy value
    }

    # Ensure all columns for features and cols_to_scale are present
    df = pd.DataFrame([input_data])

    # Ensure only required columns for scaling are scaled
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    # Ensure the DataFrame contains only the features expected by the model
    df = df[features]

    return df


def predict(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
            delinquency_ratio, credit_utilization_ratio, num_open_accounts,
            residence_type, loan_purpose, loan_type,years_at_current_address,employment_status):
    # Prepare input data
    input_df = prepare_input(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
                             delinquency_ratio, credit_utilization_ratio, num_open_accounts, residence_type,
                             loan_purpose, loan_type, years_at_current_address,employment_status)

    probability, credit_score, rating = calculate_credit_score(input_df)
    suggested_loan = suggest_loan_amount(income, loan_tenure_months)
    optimal_loan = find_optimal_loan_amount(age, income, loan_tenure_months,        avg_dpd_per_delinquency,
                                           delinquency_ratio, credit_utilization_ratio, num_open_accounts,
                                           residence_type, loan_purpose, loan_type,
                                           years_at_current_address, employment_status)
    return probability, credit_score, rating, optimal_loan,suggested_loan


def calculate_credit_score(input_df, base_score=300, scale_length=600):
    x = np.dot(input_df.values, model.coef_.T) + model.intercept_

    # Apply the logistic function to calculate the probability
    default_probability = 1 / (1 + np.exp(-x))

    non_default_probability = 1 - default_probability

    # Convert the probability to a credit score, scaled to fit within 300 to 900
    credit_score = base_score + non_default_probability.flatten() * scale_length

    # Determine the rating category based on the credit score
    def get_rating(score):
        if 300 <= score < 500:
            return 'Poor'
        elif 500 <= score < 650:
            return 'Average'
        elif 650 <= score < 750:
            return 'Good'
        elif 750 <= score <= 900:
            return 'Excellent'
        else:
            return 'Undefined'  # in case of any unexpected score

    rating = get_rating(credit_score[0])

    return default_probability.flatten()[0], int(credit_score[0]), rating


def suggest_loan_amount(income, loan_tenure_months, target_dti=0.36, annual_interest_rate=0.10):
    """
    Suggests a loan amount based on a target debt-to-income (DTI) ratio.
    """
    gross_monthly_income = income / 12
    suggested_monthly_payment = gross_monthly_income * target_dti
    
    monthly_interest_rate = annual_interest_rate / 12
    
    if monthly_interest_rate > 0:
        # Using the loan amortization formula to solve for the principal (P)
        # P = M * [(1+r)^n - 1] / [r(1+r)^n]
        n = loan_tenure_months
        r = monthly_interest_rate
        
        numerator = (1 + r)**n - 1
        denominator = r * (1 + r)**n
        
        if denominator > 0:
            suggested_loan = suggested_monthly_payment * (numerator / denominator)
            return int(suggested_loan)
    
    return 0

def find_optimal_loan_amount(age, income, loan_tenure_months, avg_dpd_per_delinquency,
                               delinquency_ratio, credit_utilization_ratio, num_open_accounts,
                               residence_type, loan_purpose, loan_type,
                               years_at_current_address, employment_status,
                               target_rating='Good'):
    """
    Finds the maximum loan amount that results in a target credit rating
    using a binary search approach.
    """
    low = 0
    high = income * 10  # A reasonable upper bound for a loan
    optimal_loan = 0

    while low <= high:
        mid = (low + high) // 2
        input_df = prepare_input(age, income, mid, loan_tenure_months, avg_dpd_per_delinquency,
                                 delinquency_ratio, credit_utilization_ratio, num_open_accounts,
                                 residence_type, loan_purpose, loan_type,
                                 years_at_current_address, employment_status)
        _, _, rating = calculate_credit_score(input_df)

        if rating == target_rating or rating == 'Excellent':
            optimal_loan = mid
            low = mid + 1
        else:
            high = mid - 1
            
    return int(optimal_loan)