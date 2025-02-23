# prompt: /content/trained_model.sav...Use the trained ML model to generate a Streamlit Web application to visualize the findings for the top 5 factors influencing the customer decision to book a flight. Should be compatible with the Streamlit community cloud platform requirements for dependencies and deployment.

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load('trained_model.sav')

# Load the dataset (replace with your actual data loading)
# Assuming the data used for training is available
try:
    df = pd.read_excel('/content/customer_booking.xlsx')
except FileNotFoundError:
    st.error("Error: customer_booking.xlsx not found. Please upload the file.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the data: {e}")
    st.stop()

# Preprocessing (same as in your training script)
le = LabelEncoder()
for col in df.columns:
  if df[col].dtype == 'object':
    df[col] = le.fit_transform(df[col])

X = df.drop('booking_complete', axis=1)
# ...

# Streamlit app
st.title("Flight Booking Prediction Dashboard")

# Display feature importances
st.header("Top 5 Factors Influencing Booking Decisions")
feature_importances = model.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False).head(5)

fig, ax = plt.subplots(figsize=(8, 6))  # Adjust figure size as needed
ax.bar(importance_df['Feature'], importance_df['Importance'])
ax.set_xlabel("Factors")
ax.set_ylabel("Importance")
ax.set_title("Top 5 Factors")
ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels
st.pyplot(fig)

# ... (Add other visualizations or interactive elements as desired)


# Example: Add a section for making predictions
# Create input fields for user to enter data
# Make prediction based on user input and display the results

# Placeholder for interactive elements.
st.header('Interactive Analysis') # Add a header

# ...(Add code for prediction or other analysis)
