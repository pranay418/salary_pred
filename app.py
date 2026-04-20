%%writefile app.py

import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os

# Define the directory where the model and encoders are saved
# This assumes they are in the same directory as app.py
model_dir = os.path.dirname(__file__)

# Load the trained Random Forest model
model_path = os.path.join(model_dir, 'random_forest_regressor_model.pkl')
model = joblib.load(model_path)

# Load the LabelEncoders for categorical features
encoders = {
    'Gender': joblib.load(os.path.join(model_dir, 'label_encoder_gender.pkl')),
    'Education Level': joblib.load(os.path.join(model_dir, 'label_encoder_education_level.pkl')),
    'Job Title': joblib.load(os.path.join(model_dir, 'label_encoder_job_title.pkl'))
}

# --- Streamlit App Interface ---
st.title('Salary Prediction App')
st.write('Enter the employee details to predict their salary.')

# Input fields for user
age = st.slider('Age', 18, 65, 30)

gender_options = encoders['Gender'].classes_
gender = st.selectbox('Gender', gender_options)

education_options = encoders['Education Level'].classes_
education_level = st.selectbox('Education Level', education_options)

job_title_options = encoders['Job Title'].classes_
job_title = st.selectbox('Job Title', job_title_options)

years_of_experience = st.slider('Years of Experience', 0.0, 40.0, 5.0, step=0.5)

# Prediction button
if st.button('Predict Salary'):
    # Preprocess user input
    gender_encoded = encoders['Gender'].transform([gender])[0]
    education_encoded = encoders['Education Level'].transform([education_level])[0]
    job_title_encoded = encoders['Job Title'].transform([job_title])[0]

    # Create a DataFrame for the model input
    input_data = pd.DataFrame([{
        'Age': age,
        'Gender': gender_encoded,
        'Education Level': education_encoded,
        'Job Title': job_title_encoded,
        'Years of Experience': years_of_experience
    }])

    # Make prediction
    predicted_salary = model.predict(input_data)[0]

    st.success(f'Predicted Salary: ${predicted_salary:,.2f}')
