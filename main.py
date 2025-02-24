# prompt: /content/cleaned_customer_booking.csv...Generate a Streamlit app that trains a ML model to determine the top 5 factors influencing the customer decision to book a flight. Should be compatible with the Streamlit community cloud platform requirements for deployment. Ensure to have relevant input elements like the CSV file upload functionality. Remember the target output column for training the ML model in the data set is labeled as booking_complete

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px


def main():
    st.title("Flight Booking Prediction")
    st.write("This app predicts the top factors influencing customer flight booking decisions.")

    # File Upload
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            # Preprocessing (add more steps if needed based on your data)
            # ... (e.g. handle missing data, convert data types, etc.)

            st.write("Data Preview:")
            st.write(df.head())

            # Feature Selection and Target Variable
            target_column = "booking_complete"  # replace with your target column name
            feature_columns = [col for col in df.columns if col != target_column]

            X = df[feature_columns]
            y = df[target_column]


            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Split data
            model = RandomForestClassifier(n_estimators=100, random_state=42) # Initialize and train the model
            model.fit(X_train, y_train)

            # Make Predictions
            y_pred = model.predict(X_test)

            # Evaluate Model
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            st.write(f"Model Accuracy: {accuracy}")
            st.write("Classification Report:\n", report)

            # Feature Importance
            feature_imp = pd.Series(model.feature_importances_, index=feature_columns).sort_values(ascending=False)
            top_5_features = feature_imp.head(5)
            st.write("Top 5 Influential Factors:")
            st.write(top_5_features)


            fig = px.bar(x=top_5_features.index, y=top_5_features.values)
            fig.update_layout(xaxis_title="Features", yaxis_title="Importance", title="Top 5 Feature Importance")
            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.info("Please upload a CSV file.")


if __name__ == "__main__":
    main()
