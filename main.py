import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import numpy as np
import base64
import time
import requests

from pathlib import Path

import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv

from config import model
from src.utils import extract_csv_content, save_uploaded_file


logo_path = Path(r"insightify_logo.png")

# Function to load and encode the image to base64
def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# Add custom favicon to the tab (using base64-encoded image)
if logo_path.exists():
    favicon_base64 = encode_image(logo_path)
    st.markdown(
        f"""
        <head>
            <link rel="icon" href="data:image/png;base64,{favicon_base64}" type="image/png">
        </head>
        """,
        unsafe_allow_html=True
    )
else:
    st.warning("Logo file not found. Ensure the path is correct.")


# Add the logo next to the title
if logo_path.exists():
    st.markdown(
        f"""
        <div class="header-logo">
            <img src="data:image/png;base64,{encode_image(logo_path)}" alt="Insightify Logo">
        </div>
        """,
        unsafe_allow_html=True
    )
    

# Add custom styles for white background, dark blue text, and light blue upload area
st.markdown(
    """
   <style>
        

        div[data-testid="stFileUploader"] label {
            color: white !important; /* Ensures label text is white for file uploaders */
        }

        div[data-testid="stUploadedFile"] {
            color: white !important; /* Ensures the text for uploaded file names is white */
        }

        a.css-1ekf3qb {
            color: white !important; /* Ensures download link text is white */
        }
        
        div.stMarkdown, div.stText, div[data-testid="stDataFrame"] {
            color: white !important; /* Sets text color to white */
            font-family: Arial, sans-serif; /* Optional: Change font-family */
        }

        div[data-baseweb="select"] {
            color: white !important; /* Ensures dropdown text is white */
        }

        div[data-baseweb="select"] .css-1wa3eu0 {
            color: white !important; /* Makes dropdown placeholder text white */
        }

        div[data-testid="stVerticalBlock"] label {
            color: white !important; /* Ensures labels are white */
        }

        div.stMarkdown, div.stText, div[data-testid="stDataFrame"] {
            color: white !important; /* Sets text color to white */
            font-family: Arial, sans-serif; /* Change font-family if needed */
        }
        
        body {
            background: #1E90FF !important; /* Gradient blue */
            color: #ffffff !important; /* White font for text */
        }

        /* Adjust Streamlit app's container with the same gradient */
        .stApp {
            background: #1E90FF !important; /* Gradient blue */
            overflow: hidden;
        }
        
        /* Logo animation */
        .header-logo img {
            animation: fadeInZoom 3s ease-in-out infinite alternate;
        }
        @keyframes fadeInZoom {
            from {
                opacity: 0.5;
                transform: scale(1);
            }
            to {
                opacity: 1;
                transform: scale(1.1);
            }
        }

        /* Button hover animation */
        button {
            background-color: #80c1ff !important;
            color: #ffffff !important;
            border-radius: 10px !important;
            font-weight: bold !important;
            animation: pulse 2s infinite;
        }
        button:hover {
            background-color: #003366 !important;
            color: #ffffff !important;
        }
        @keyframes pulse {
            0% {
                transform: scale(1);
            }
            50% {
                transform: scale(1.05);
            }
            100% {
                transform: scale(1);
            }
        }

        /* Add subtle animations to text */
        h1, h2, h3 {
            animation: slideIn 1.5s ease-in-out;
        }
        @keyframes slideIn {
            from {
                transform: translateX(-50%);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }

        /* Add a logo and button container */
        .header-container {
            display: flex;
            flex-direction: column; /* Stack vertically */
            align-items: center; /* Center items horizontally */
            gap: 10px; /* Adjust the space between logo and buttons */
            padding: 10px;
        }

        /* Logo styling */
       .header-logo {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            font-family: Arial, sans-serif;
            color: #ffffff;
            font-weight: bold;
            font-size: 24px;
            margin-top: 25px;
        }

        .header-logo img {
            height: 350px; /* Adjust logo size */
            margin-bottom: -100px;
        }

        /* Customize the drag-and-drop upload box */
        .stFileUploader {
            background-color: #ffffff !important; /* White background */
            border: 2px solid #003366 !important; /* Dark blue border */
            border-radius: 10px; /* Rounded corners */
            color: #003366 !important; /* Dark blue text */
        }

        /* Customize the "Drag and drop file here" text */
        .stFileUploader div {
            color: #003366 !important; /* Dark blue font */
            font-weight: bold !important;
        }

        /* Customize file uploader header ("Upload a CSV file...") */
        div[data-testid="stFileUploader"] > label {
            color: #ffffff !important; /* Dark blue text */
            font-weight: bold !important; /* Bold text for emphasis */
        }

        /* Customize input field header ("Ask questions with Insightify") */
        div[data-testid="stTextInput"] > label {
            color: #ffffff !important; /* Dark blue font */
            font-weight: bold !important; /* Bold text for emphasis */
        }

        /* Style navigation bar */
        header, footer {
            background: #1E90FF !important; /* Gradient blue */
            color: #003366 !important; /* Dark blue font */
            border-bottom: 1px solid #80c1ff !important; /* Light blue border */
        }

        /* Customize buttons */
        button {
            background-color: #1E90FF !important; /* Dark blue background */
            color: #ffffff !important; /* White text */
            border-radius: 10px !important;
            font-weight: bold !important;
        }

        /* Customize "Browse files" button in file uploader */
        button[aria-label="Browse files"] {
            background-color: #80c1ff !important; /* Light blue button */
            color: #ffffff !important; /* White text */
            font-weight: bold !important; /* Bold font for emphasis */
        }

        /* Optional: Style the header and section titles */
        h1, h2, h3, h4, h5, h6 {
            color: #003366 !important; /* Dark blue headers */
        }

        /* Input field styling */
        textarea, input {
            background-color:  #ffffff !important; /* Light blue input background */
            color:  #000000 !important; /* Dark blue font */
            border-radius: 10px !important; /* Rounded corners */
            font-size: 14px !important; /* Set the font size */
        }
        
    </style>

    """,
    unsafe_allow_html=True
)




# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("AIzaSyAXPS3N96tyjQ3SaQgx_pgRDE_o1YbPYkk"))

def generate_visualizations_with_gemini(cleaned_df):
    """
    Generates visualizations with insights and data reports for the cleaned dataset based on user selection.

    Args:
        cleaned_df (pd.DataFrame): The cleaned dataset.

    Returns:
        None
    """
    st.markdown('<h3 style="color:white;">Choose a Visualization and View Insights with a Data Report</h3>', unsafe_allow_html=True)

    # Select visualization type
    visualization_type = st.selectbox(
        "Select the type of visualization",
        ["Histogram", "Line Chart", "Box Plot", "Stacked Line Chart", "Gauge Chart"]
    )

    # Select columns for visualization
    numeric_columns = cleaned_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    column_options = st.multiselect(
        "Select columns for visualization (Choose 1 or more)",
        options=numeric_columns,
        default=numeric_columns[:1]
    )

    if not column_options:
        st.error("Please select at least one column.")
        return

    # Generate insights, visualizations, and reports
    st.write(f"### {visualization_type} Visualization, Insights, and Data Report")
    try:
        if visualization_type == "Histogram":
            for col in column_options:
                mean_value = cleaned_df[col].mean()
                median_value = cleaned_df[col].median()
                skewness = cleaned_df[col].skew()
                st.write(f"**Data Report for Column: {col}**")
                st.write(
                    f"- Mean: {mean_value:.2f}\n"
                    f"- Median: {median_value:.2f}\n"
                    f"- Skewness: {skewness:.2f} (Indicates {'right' if skewness > 0 else 'left'} skew)\n"
                    f"- Standard Deviation: {cleaned_df[col].std():.2f}\n"
                    f"- Minimum Value: {cleaned_df[col].min():.2f}\n"
                    f"- Maximum Value: {cleaned_df[col].max():.2f}"
                )
                # Use Plotly for interactive histogram
                fig = px.histogram(cleaned_df, x=col, nbins=20, title=f"Histogram for {col}")
                fig.update_layout(bargap=0.2)
                st.plotly_chart(fig)

        elif visualization_type == "Line Chart":
            if len(column_options) < 2:
                st.error("Please select at least two columns for a line chart.")
            else:
                selected_columns_df = cleaned_df[column_options]
                fig = px.line(selected_columns_df, title="Line Chart")
                st.plotly_chart(fig)

        elif visualization_type == "Box Plot":
            for col in column_options:
                fig = px.box(cleaned_df, y=col, title=f"Box Plot for {col}")
                st.plotly_chart(fig)

        elif visualization_type == "Stacked Line Chart":
            if len(column_options) < 2:
                st.error("Please select at least two columns for a stacked line chart.")
            else:
                stacked_data = cleaned_df[column_options].cumsum()
                fig = px.area(stacked_data, title="Stacked Line Chart", labels={"value": "Cumulative Value"})
                st.plotly_chart(fig)

        elif visualization_type == "Gauge Chart":
            col = column_options[0]
            gauge_value = cleaned_df[col].mean()
            max_value = cleaned_df[col].max()
            min_value = cleaned_df[col].min()

            st.write(f"**Data Report for Gauge Chart: {col}**")
            st.write(
                f"- Column: {col}\n"
                f"- Average Value: {gauge_value:.2f}\n"
                f"- Max Value: {max_value:.2f}\n"
                f"- Min Value: {min_value:.2f}"
            )

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=gauge_value,
                title={'text': f"{col} Gauge Chart"},
                gauge={
                    'axis': {'range': [min_value, max_value]},
                    'bar': {'color': "blue"},
                    'steps': [
                        {'range': [min_value, (min_value + max_value) / 2], 'color': "lightgray"},
                        {'range': [(min_value + max_value) / 2, max_value], 'color': "lightblue"}
                    ]
                }
            ))
            st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Error generating visualization: {e}")

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
            # Categorical features
            if df[col].dtype == "object":
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
            # Numerical features
            elif df[col].dtype in ["int64", "float64"]:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)

    # Handle outliers using IQR
    for col in df.select_dtypes(include=["int64", "float64"]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

    # Convert object columns to proper data types if necessary
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                pass

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    return df

def main():
    st.title("")

    # File upload section
    uploaded_file = st.file_uploader("Upload a CSV file for visualization", type="csv")
    save_directory = "data"
    os.makedirs(save_directory, exist_ok=True)

    if uploaded_file is not None:
        # Save and clean the uploaded file
        saved_file_path = save_uploaded_file(uploaded_file, save_directory)
        raw_df = pd.read_csv(saved_file_path)
        cleaned_df = clean_data(raw_df)

        # Display the cleaned dataset
        #st.write("### Cleaned Data Preview")
        #st.dataframe(cleaned_df.head())

        # Generate visualizations with user input
        generate_visualizations_with_gemini(cleaned_df)


    # Gemini AI Question-Answering
    user_question = st.text_input("Ask questions with Insightify")
    if user_question and uploaded_file is not None:
        chat_session = model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": extract_csv_content(saved_file_path),
                }
            ]
        )
        response = chat_session.send_message(user_question)
        st.write(response.text)


if __name__ == "__main__":
    main()
