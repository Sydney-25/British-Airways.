# prompt: Show the dashboard generated using a Streamlit app that is compatible with the Streamlit Community Cloud platform requirements and dependencies.

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Streamlit app
st.title("Customer Booking Analysis")

# File upload
uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        st.write("Data Preview:")
        st.dataframe(df.head())

        # Data Cleaning (add more data cleaning steps as needed)
        for col in df.select_dtypes(include=['number']):
            df[col].fillna(df[col].mean(), inplace=True)
        df.drop_duplicates(inplace=True)

        # Feature Engineering (add any new features as needed)
        # ...

        # Preprocessing: Label Encoding for categorical columns
        le = LabelEncoder()
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = le.fit_transform(df[col])

        # Prepare data for the model
        X = df.drop('booking_complete', axis=1)
        y = df['booking_complete']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model training
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Model Accuracy: {accuracy}")

        # Feature Importance
        feature_importances = model.feature_importances_
        importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        # Plotting
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        axes[0].bar(importance_df['Feature'].head(5), importance_df['Importance'].head(5))
        axes[0].set_title('Top 5 Factors Influencing Booking Decisions')
        axes[0].set_xlabel('Factors')
        axes[0].set_ylabel('Importance')
        axes[0].tick_params(axis='x', rotation=45)

        axes[1].text(0.5, 0.5, f'Model Accuracy: {accuracy:.2f}', ha='center', va='center', fontsize=14)
        axes[1].set_title('Model Performance')
        axes[1].axis('off')  # Turn off axis for text-only plot

        plt.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred: {e}")
