# Customer Booking Analysis Dashboard

An interactive web application that leverages machine learning to analyze customer booking data and predict booking completion. Built with Streamlit, the tool processes data from an Excel file to train a Random Forest Classifier, providing insights into feature importance and model accuracy.

## Table of Contents

*   [Overview](#overview)
*   [Features](#features)
*   [Installation](#installation)
*   [Usage](#usage)
*   [Project Structure](#project-structure)
*   [Customization](#customization)
*   [Contributing](#contributing)
*   [Contact](#contact)

## Overview

The Customer Booking Analysis Dashboard is designed to help analyze factors influencing booking completion. By uploading customer booking data in XLSX format, the application:

*   Processes and prepares the data by handling missing values and duplicates.
*   Trains a Random Forest Classifier to predict booking completion.
*   Visualizes the top factors influencing booking decisions based on feature importance.
*   Displays the model's accuracy.

## Features

*   **Data Upload:** Easily upload your customer booking data (XLSX format).
*   **Data Cleaning:** Handles missing values in numerical columns by filling them with the mean, and removes duplicate rows.
*   **Label Encoding:** Converts categorical columns to numerical data using Label Encoding.
*   **Model Training:** Trains a Random Forest Classifier model.
*   **Model Evaluation:** Calculates and displays the model's accuracy on a test dataset.
*   **Feature Importance Visualization:** Displays a bar chart of the top 5 features influencing booking decisions.
*   **Model Performance Display:** Displays the model's accuracy in a text format on the plot.

## Installation

1.  **Clone the Repository:**

    ```bash
    git clone <your_repository_url>
    cd <your_repository_directory>
    ```

2.  **Create and Activate a Virtual Environment** (optional but recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Ensure that you have installed the following libraries:**

    *   Streamlit
    *   Pandas
    *   Scikit-learn (sklearn)
    *   Matplotlib

## Usage

1.  **Start the Application:**

    ```bash
    streamlit run main.py
    ```

2.  **Upload Your Data:**

    *   Click the "Choose an Excel file" button and select your XLSX file. The Excel file *must* have a column named `booking_complete` representing the target variable.

3.  **Model Training & Evaluation:**

    *   Once the XLSX file is uploaded, the application processes the data, trains the model, and evaluates its accuracy.
    *   The dashboard displays the model accuracy and a visualization of feature importance.

## Project Structure
customer-booking-analysis/
├── main.py # Main application file for running the Streamlit app
├── README.md # Project documentation (this file)
├── requirements.txt # List of Python dependencies

## Customization

*   **Data Cleaning:** Enhance the data cleaning process in `main.py` by adding more robust checks for missing values, handling outliers, and addressing data inconsistencies specific to your dataset.
*   **Feature Engineering:** Implement feature engineering techniques in `main.py` to create new features from existing ones. This is currently a placeholder; add your domain-specific knowledge to create useful features.
*   **Model Selection:** Experiment with different classification models from `sklearn` (e.g., Logistic Regression, Support Vector Machines) in `main.py` to see which performs best.
*   **Hyperparameter Tuning:** Tune the hyperparameters of the Random Forest model (or any chosen model) using techniques like GridSearchCV or RandomizedSearchCV to optimize performance. Add code for this in `main.py`.
*   **Visualization:** Customize the plots in `main.py` using `matplotlib` to display different aspects of the data or model results.  Consider adding interactive elements.

## Contributing

Contributions are welcome! To contribute:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature`).
3.  Commit your changes.
4.  Push the branch and open a pull request.

## Contact

For questions or suggestions related to this project, please feel free to reach out by creating an issue in this GitHub repository.
