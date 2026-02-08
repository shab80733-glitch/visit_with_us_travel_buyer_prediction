import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the trained model
model_path = hf_hub_download(repo_id="ShabN/visit-with-us-travel-buyer-prediction", filename="best_tourism_model.joblib")
model = joblib.load(model_path)

# Streamlit UI
st.title("Tourism Buyer Prediction App")
st.icon = "✈️"
st.write("Welcome to the Tourism Buyer Prediction App!")
st.write("Enter customer details to predict whether they will purchase the Wellness Tourism Package.")

#User input
Age = st.number_input('Age', min_value=18, max_value=100, value=30),
City_Tier = st.selectbox('CityTier', ['Tier 1', 'Tier 2', 'Tier 3']),
Duration_Of_Pitch = st.number_input('DurationofPitch', min_value=1, max_value=100, value=30),
Number_Of_Person_Visiting = st.number_input('NumberofPersonVisiting', min_value=1, max_value=10, value=2),
Number_Of_Followups = st.number_input('NumberofFollowups', min_value=1, max_value=10, value=2),
Preferred_Property_Star= st.selectbox('PreferredPropertyStar', [1, 2, 3, 4, 5]),
Number_Of_Trips = st.number_input('NumberOfTrips', min_value=1, max_value=10, value=2),
Passport = st.selectbox('Passport', [0, 1]),
Pitch_Satisfaction_Score = st.number_input('PitchSatisfactionScore', min_value=1, max_value=5, value=3),
Own_Car = st.selectbox('OwnCar', [0, 1]),
Number_Of_Children_Visiting = st.number_input('NumberOfChildrenVisiting', min_value=0, max_value=10, value=0),
Monthly_Income= st.number_input('MonthlyIncome', min_value=0, max_value=100000, value=50000),
Type_of_Contact = st.selectbox('TypeofContact', ['Company Invited', 'Self Inquiry']),
Occupation = st.selectbox('Occupation', ['Salaried', 'Free Lancer','Small Business', 'Large Business' ])
Designation = st.selectbox('Designation', ['Executive', 'Manager', 'Senior Manager', 'AVP', 'VP']),
Gender = st.selectbox('Gender', ['Male', 'Female']),
Marital_Status = st.selectbox('MaritalStatus', ['Single', 'Married', 'Divorced']),
Product_Pitched = st.selectbox('ProductPitched', ['Basic', 'Deluxe', 'Standard', 'Super Deluxe', 'King'])

# Assemble input into DataFrame
'Age': Age
'CityTier': City_Tier
'DurationOfPitch': Duration_Of_Pitch
'NumberOfPersonVisiting': Number_Of_Person_Visiting
'NumberOfFollowups': Number_Of_Followups
'PreferredPropertyStar': Preferred_Property_Star
'NumberOfTrips': Number_Of_Trips
'Passport': Passport
'PitchSatisfactionScore': Pitch_Satisfaction_Score
'OwnCar': Own_Car
'NumberOfChildrenVisiting': Number_Of_Children_Visiting
'MonthlyIncome': Monthly_Income
'TypeofContact': Type_of_Contact
'Occupation': Occupation
'Designation': Designation
'Gender': Gender
'MaritalStatus': Marital_Status
'ProductPitched': Product_Pitched

#Predict Button
if st.button('Predict'):
  prediction = model.predict(input_df)
  st.write(f"Prediction: {prediction[0]}")
  if prediction[0] == 1:
    st.write("The customer is likely to purchase the Wellness Tourism Package.")
