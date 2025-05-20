import streamlit as st
import pandas as pd
import joblib
import time
import numpy as np
from sklearn.preprocessing import StandardScaler

st.write("""
# Customer Churn Prediction App

This app predicts whether a customer will **churn** or not!
""")

# Load your pre-trained model and scaler
@st.cache_data
def load_artifacts():
    model = joblib.load('rfc_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_artifacts()

st.sidebar.header('User Input Parameters')

def user_input_features():
    credit_score = st.sidebar.slider('Credit Score', 300, 850, 650)
    age = st.sidebar.slider('Age', 18, 100, 35)
    tenure = st.sidebar.slider('Tenure (years with bank)', 0, 10, 2)
    balance = st.sidebar.number_input('Account Balance', 0.0, 300000.0, 10000.0)
    num_products = st.sidebar.slider('Number of Products', 1, 4, 1)
    
    gender = st.sidebar.radio('Gender', ['Female', 'Male'])
    has_cr_card = st.sidebar.radio('Has Credit Card?', ['Yes', 'No'])
    is_active = st.sidebar.radio('Is Active Member?', ['Yes', 'No'])
    geography = st.sidebar.selectbox('Geography', ['France', 'Germany', 'Spain'])
    
    estimated_salary = st.sidebar.number_input('Estimated Salary', 0.0, 200000.0, 50000.0)
    
    # Convert to numerical
    gender = 0 if gender == 'Female' else 1
    has_cr_card = 1 if has_cr_card == 'Yes' else 0
    is_active = 1 if is_active == 'Yes' else 0
    
    # One-hot encode geography
    geo_france = 1 if geography == 'France' else 0
    geo_germany = 1 if geography == 'Germany' else 0
    geo_spain = 1 if geography == 'Spain' else 0
    
    data = {
        'CreditScore': credit_score,
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': has_cr_card,
        'IsActiveMember': is_active,
        'EstimatedSalary': estimated_salary,
        'Geography_France': geo_france,
        'Geography_Germany': geo_germany,
        'Geography_Spain': geo_spain
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

expected_columns = [
    'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 
    'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
    'Geography_France', 'Geography_Germany', 'Geography_Spain'
]

df = df[expected_columns]

input_array = df.values.astype(np.float64)

try:
    scaled_features = scaler.transform(input_array)
    prediction = model.predict(scaled_features)
    prediction_proba = model.predict_proba(scaled_features)
    
    st.subheader('Prediction')
    st.write('Will the customer churn?' )
    st.write('Yes' if prediction[0] == 1 else 'No')

    st.subheader('Prediction Probability')
    st.write(f'Probability of churning: {prediction_proba[0][1]:.2%}')
    st.write(f'Probability of staying: {prediction_proba[0][0]:.2%}')

except Exception as e:
    st.error(f"An error occurred during prediction: {str(e)}")
    st.write("Please check that your input data matches the model's expectations")

my_bar = st.progress(0)
for percent_complete in range(100):
    time.sleep(0.05)
    my_bar.progress(percent_complete + 1)