# prompt: Make the Streamlit app to be compatible with the Streamlit community cloud platform requirements for the deployment 

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the trained model
try:
    model = joblib.load('trained_model.sav')
except FileNotFoundError:
    st.error("Trained model file not found. Please upload the 'trained_model.sav' file.")
    st.stop()  # Stop execution if the model file is not found
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()

# Streamlit app
st.title("Customer Booking Prediction App")

uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

if uploaded_file is not None:
    try:
        # Read the uploaded file
        df = pd.read_excel(uploaded_file)

        # Preprocessing (adapt as needed for your specific features)
        def preprocess_input(input_df):
            le = LabelEncoder()
            for col in input_df.columns:
                if input_df[col].dtype == 'object':
                    input_df[col] = le.fit_transform(input_df[col])
            return input_df
        
        # Example usage (replace with your actual features and preprocessing):
        st.subheader("Enter Feature Values")
        num_adults = st.number_input("Number of Adults", min_value=0)
        num_children = st.number_input("Number of Children", min_value=0)
        num_weekend_nights = st.number_input("Number of Weekend Nights", min_value=0)
        # ... add input fields for all the other features your model expects


        if st.button("Predict"):
              # Create a DataFrame from user input
              user_input_df = pd.DataFrame({
                  'num_adults': [num_adults],
                  'num_children': [num_children],
                  'num_weekend_nights': [num_weekend_nights],
                  # ... Add the other user inputs to the DataFrame
              })


              user_input_df = preprocess_input(user_input_df)


              # Ensure the user input has the same columns as the training data
              missing_cols = set(X.columns) - set(user_input_df.columns)  # Assuming X from training is available
              for c in missing_cols:
                  user_input_df[c] = 0 # Or handle missing values appropriately
              user_input_df = user_input_df[X.columns] # Ensure same column order

              prediction = model.predict(user_input_df)

              if prediction[0] == 1:
                  st.success("Booking is likely to be completed.")
              else:
                  st.error("Booking is unlikely to be completed.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
