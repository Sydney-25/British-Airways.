import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import io  # Import the io module

# Streamlit app
st.title("Customer Booking Analysis")

# Make the layout more mobile-friendly
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        max-width: 100%;
        padding: 1rem;
    }
    [data-testid="stHeader"] {
        padding: 0;
    }
    [data-testid="stToolbar"] {
        right: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# File upload
uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

if uploaded_file is not None:
    try:
        # Use BytesIO to handle the file in memory
        excel_data = io.BytesIO(uploaded_file.read())
        df = pd.read_excel(excel_data)  # Read from the BytesIO object

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
        fig, axes = plt.subplots(1, 1, figsize=(8, 6)) # Adjusted for single column

        axes.bar(importance_df['Feature'].head(5), importance_df['Importance'].head(5))
        axes.set_title('Top 5 Factors Influencing Booking Decisions')
        axes.set_xlabel('Factors')
        axes.set_ylabel('Importance')
        axes.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        st.pyplot(fig)

        st.write(f"Model Accuracy: {accuracy:.2f}") # Accuracy moved to a st.write

    except Exception as e:
        st.error(f"An error occurred: {e}")
