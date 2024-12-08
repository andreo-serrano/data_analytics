import csv
import os
import time

import pandas as pd
import streamlit as st
import numpy as np
from sklearn.impute import KNNImputer
from scipy.stats import zscore

# Custom CSS for white font
def add_custom_css():
    st.markdown(
        """
        <style>
        div.stMarkdown, div.stText, div[data-testid="stDataFrame"] {
            color: white !important; /* Sets text color to white */
            font-family: Arial, sans-serif; /* Change font-family if needed */
        }
        body {
            background-color: #1e1e1e; /* Optional: Dark background for contrast */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def extract_csv_content(pathname: str) -> list[str]:
    """
    Extracts the content of a CSV file and returns it as a list of strings.

    Args:
        pathname (str): The path to the CSV file.

    Returns:
        list[str]: A list containing the content of the CSV file, with start and end indicators.
    """
    parts = [f"--- START OF CSV {pathname} ---"]
    with open(pathname, "r", newline="") as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            parts.append(" ".join(row))
    parts.append(f"--- END OF CSV {pathname} ---")
    return parts


def save_uploaded_file(uploaded_file, save_directory):
    """
    Saves the uploaded file to the specified directory.

    Args:
        uploaded_file (streamlit.uploadedfile.UploadedFile): The uploaded file.
        save_directory (str): The directory where the file will be saved.

    Returns:
        str: The path where the file is saved.
    """
    file_name = "file.csv"  # You might want to customize the file name
    file_path = os.path.join(save_directory, file_name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    saved_file_path = os.path.join(save_directory, file_name)
    success_message = st.success("File saved successfully!")
    time.sleep(1)  # Wait for 1 second (change if needed)
    success_message.empty()  # Empty the success message

    # Create a DataFrame from the saved CSV file
    df = pd.read_csv(file_path)

    # Display the first few rows of the DataFrame
    st.write("Uncleaned DataFrame:")
    st.dataframe(df)
    
    # Clean the data
    cleaned_df = clean_data(df)

    # Display the first few rows of the cleaned DataFrame
    st.write("Cleaned DataFrame:")
    st.dataframe(cleaned_df)

    return saved_file_path

def clean_data(df):
    """
    Cleans the DataFrame by handling missing values, outliers, inconsistent data types, and duplicate rows.

    Args:
        df (pandas.DataFrame): The DataFrame to clean.

    Returns:
        pandas.DataFrame: The cleaned DataFrame.
    """

    # Handle missing values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            # Categorical features:
            if df[col].dtype == 'object':
                # Consider mode imputation for frequent categories:
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)

            # Numerical features:
            elif df[col].dtype == 'int64' or df[col].dtype == 'float64':
                # Use median for robust imputation, especially in skewed distributions:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)

                # For time series data, consider interpolation or forecasting techniques.

    # Handle outliers
    # You can use statistical methods (e.g., Z-score, IQR) or domain knowledge to identify outliers.
    # Here's a basic example using IQR:
    for col in df.columns:
        if df[col].dtype == 'int64' or df[col].dtype == 'float64':
            # Calculate quartiles and IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            # Define outlier thresholds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Winsorize outliers
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])


    # Handle inconsistent data types
    for col in df.columns:
        if df[col].dtype == 'object':
            # Convert to appropriate data type if needed (e.g., datetime)
            # Example:
            # df[col] = pd.to_datetime(df[col], format='%Y-%m-%d')
            pass
        elif df[col].dtype == 'int64' or df[col].dtype == 'float64':
            # Check for non-numeric values and handle accordingly
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col].fillna(df[col].mean(), inplace=True)  # Fill missing values after conversion

    # Handle duplicate rows
    df.drop_duplicates(inplace=True)

    return df
