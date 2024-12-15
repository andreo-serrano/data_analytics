import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.animation as animation
import seaborn as sns
import numpy as np
import base64
import time
import plotly.graph_objects as go
import requests

from pathlib import Path
import plotly.io as pio
import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime

from config import model
from src.utils import extract_csv_content, save_uploaded_file

from io import BytesIO
from fpdf import FPDF

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

def generate_full_pdf(columns, cleaned_df, visualization_type):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    try:
        if visualization_type == "Box Plot":
            for col in columns:
                # Add a new page for each column
                pdf.add_page()
                pdf.set_font("Arial", style="B", size=14)
                pdf.cell(200, 10, txt=f"Box Plot for Column: {col}", ln=True, align="C")
                pdf.ln(10)

                # Generate and save the box plot
                fig, ax = plt.subplots()
                sns.boxplot(y=cleaned_df[col], ax=ax)
                ax.set_title(f"Box Plot for {col}")

                # Save the chart to a buffer
                buf = BytesIO()
                fig.savefig(buf, format="png")
                plt.close(fig)
                buf.seek(0)

                # Add the chart image to the PDF
                img_path = f"{col}_boxplot.png"
                with open(img_path, "wb") as f:
                    f.write(buf.getbuffer())
                pdf.image(img_path, x=10, y=pdf.get_y(), w=190)
                os.remove(img_path)  # Cleanup the image file after adding to PDF

                # Generate Gemini API insights and add them below the chart in the PDF
                # prompt = f"Provide insights for a box plot visualization of the column '{col}', highlighting outliers and the spread of the data."
                # ai_insight = get_dynamic_insights_gemini(prompt, model)

                # pdf.ln(50)  # Add some space after the chart
                # pdf.set_font("Arial", size=12)
                # pdf.multi_cell(0, 10, f"Gemini Insight: {ai_insight}")

                # Move to the next line for the timestamp
                pdf.ln(10)

                # Add the timestamp at the very top right
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                pdf.set_y(10)  # Move the cursor to the top
                pdf.set_x(200 - 10 - pdf.get_string_width(timestamp))  # Align to the right
                pdf.set_font("Arial", size=10)
                pdf.cell(0, 10, f"Report generated on: {timestamp}", ln=True, align="R")



        elif visualization_type == "Histogram":
            for col in columns:
                # Calculate statistics for the histogram
                mean = cleaned_df[col].mean()
                median = cleaned_df[col].median()
                std_dev = cleaned_df[col].std()

                # Add a page for each column
                pdf.add_page()
                pdf.set_font("Arial", style="B", size=14)
                pdf.cell(200, 10, txt=f"Histogram for Column: {col}", ln=True, align="C")
                pdf.ln(10)

                # Add statistics to the PDF
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, txt=f"Mean: {mean:.2f}", ln=True)
                pdf.cell(200, 10, txt=f"Median: {median:.2f}", ln=True)
                pdf.cell(200, 10, txt=f"Standard Deviation: {std_dev:.2f}", ln=True)
                pdf.ln(10)

                # Generate and save the histogram
                fig, ax = plt.subplots()
                sns.histplot(cleaned_df[col], bins=20, kde=True, ax=ax, color="#636EFA")
                ax.set_title(f"Histogram for {col}")

                # Save the histogram to a buffer
                buf = BytesIO()
                fig.savefig(buf, format="png")
                plt.close(fig)
                buf.seek(0)

                # Add the chart image to the PDF
                img_path = f"{col}_histogram.png"
                with open(img_path, "wb") as f:
                    f.write(buf.getbuffer())
                pdf.image(img_path, x=10, y=pdf.get_y(), w=190)
                os.remove(img_path)

                # Add the timestamp at the very top right
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                pdf.set_y(250)  # Move the cursor to the top
                pdf.set_x(200 - 10 - pdf.get_string_width(timestamp))  # Align to the right
                pdf.set_font("Arial", size=10)
                pdf.cell(0, 10, f"Report generated on: {timestamp}", ln=True, align="R")


        elif visualization_type == "Line Chart":
            pdf.add_page()
            pdf.set_font("Arial", style="B", size=14)
            pdf.cell(200, 10, txt="Line Chart for Selected Columns", ln=True, align="C")
            pdf.ln(10)

            # Filter and plot all selected columns
            fig, ax = plt.subplots()
            for col in columns:
                ax.plot(cleaned_df.index, cleaned_df[col], label=col)
            ax.set_title("Line Chart for Selected Columns")
            ax.set_xlabel("Index")
            ax.set_ylabel("Values")
            ax.legend()
            fig.tight_layout()

            # Save the chart to a buffer
            buf = BytesIO()
            fig.savefig(buf, format="png")
            plt.close(fig)
            buf.seek(0)

            # Add the chart image to the PDF
            img_path = "line_chart.png"
            with open(img_path, "wb") as f:
                f.write(buf.getbuffer())
            pdf.image(img_path, x=10, y=pdf.get_y(), w=190)
            os.remove(img_path)

        elif visualization_type == "Stacked Line Chart":
            pdf.add_page()
            pdf.set_font("Arial", style="B", size=14)
            pdf.cell(200, 10, txt="Stacked Line Chart", ln=True, align="C")
            pdf.ln(10)

            # Generate the stacked line chart
            fig = px.area(cleaned_df[columns], title="Stacked Line Chart", labels={"index": "Index"})

            # Save the chart to a buffer
            buf = BytesIO()
            fig.write_image(buf, format="png")
            buf.seek(0)

            # Add the chart image to the PDF
            img_path = "stacked_line_chart.png"
            with open(img_path, "wb") as f:
                f.write(buf.getbuffer())
            pdf.image(img_path, x=10, y=pdf.get_y(), w=190)

    except Exception as e:
        st.error(f"Error generating visualization: {e}")

    return pdf

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("AIzaSyDQzoYfwL-1H9bE_tYez3-2-yoCIXi_Cn8"))

# Theme for Seaborn
sns.set_theme(style="whitegrid", palette="pastel")

def generate_visualizations_with_gemini(cleaned_df):
    """
    Generate various data visualizations and provide actionable insights using Gemini API.
    Args:
        cleaned_df (pd.DataFrame): The cleaned dataset to visualize.
    """
    from config import model  # Import the configured model from config.py
    
    st.markdown('<h3 style="color:white;">Choose a Visualization and View Insights with a Data Report</h3>', unsafe_allow_html=True)
    
    visualization_type = st.selectbox(
        "Select the type of visualization",
        ["Histogram", "Line Chart", "Box Plot", "Stacked Line Chart", "Gauge Chart"]
    )
    
    numeric_columns = cleaned_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    column_options = st.multiselect(
        "Select columns for visualization (Choose 1 or more)",
        options=numeric_columns,
        default=numeric_columns[:1]
    )
    
    if not column_options:
        st.error("Please select at least one column.")
        return
    
    st.write(f"### {visualization_type} Visualization, Insights, and Data Report")

    if visualization_type:
        pdf = generate_full_pdf(column_options, cleaned_df, visualization_type)
        # Provide a download link for the generated PDF
        st.download_button(
            label="Download PDF",
            data=pdf.output(dest="S").encode("latin1"),
            file_name=f"{visualization_type}_visualization_report.pdf",
            mime="application/pdf"
        )
    
    try:
        for col in column_options:
            st.write(f"#### **Column: {col}**")
            
            if cleaned_df[col].isnull().any():
                missing_percentage = cleaned_df[col].isnull().mean() * 100
                st.warning(f"Column '{col}' has {missing_percentage:.2f}% missing values. Consider handling missing data.")
            
            if visualization_type == "Histogram":
                generate_histogram_with_gemini_insights(cleaned_df, col, model)
            elif visualization_type == "Line Chart":
                generate_line_chart_with_gemini_insights(cleaned_df, column_options, model)
                break
            elif visualization_type == "Box Plot":
                generate_box_plot_with_gemini_insights(cleaned_df, col, model)
            elif visualization_type == "Stacked Line Chart":
                generate_stacked_line_chart_with_gemini_insights(cleaned_df, column_options, model)
                break
            elif visualization_type == "Gauge Chart":
                generate_gauge_chart_with_gemini_insights(cleaned_df, col, model)
    except Exception as e:
        st.error(f"Error generating visualization: {e}")

def get_dynamic_insights_gemini(prompt, model):
    """
    Generate dynamic insights using the Gemini API via the chat session model in config.py.
    Args:
        prompt (str): The prompt describing the data and visualization.
        model (GenerativeModel): The pre-configured generative model from config.py.
    Returns:
        str: Generated insight from the Gemini API.
    """
    try:
        # Start a chat session with the model
        chat_session = model.start_chat(
            history=[{"role": "user", "parts": ["The user uploaded a dataset."]}]
        )
        # Send the prompt and get the response
        response = chat_session.send_message(prompt)
        return response.text
    except Exception as e:
        return f"Unable to generate insights: {e}"

# Modular functions for visualizations with Gemini API integration
def generate_histogram_with_gemini_insights(df, col, model):
    """
    Generate a histogram with dynamic insights from Gemini API.
    Args:
        df (pd.DataFrame): The dataset.
        col (str): Column to visualize.
        model (GenerativeModel): The configured Gemini model.
    """
    st.write("A **Histogram** is a graphical representation of the distribution of a dataset. "
            "It organizes the data into bins (intervals) and displays the frequency of data points within each bin. "
            "This visualization is useful for understanding the shape, spread, and central tendency of the data.")
    st.write(f"- Mean: {df[col].mean():.2f}")
    st.write(f"- Median: {df[col].median():.2f}")
    st.write(f"- Std Dev: {df[col].std():.2f}")
    if df[col].skew() > 1:
        st.warning("The data is highly skewed. Consider transforming it for better analysis.")
    
    fig = px.histogram(df, x=col, nbins=20, title=f"Histogram for {col}", marginal="box", color_discrete_sequence=["#636EFA"])
    st.plotly_chart(fig, use_container_width=True)
    
    # Generate Gemini API insights
    prompt = f"Provide insights for a histogram visualization of the column and do not provide any negative thoughts about the dataset '{col}' with the following stats: mean={df[col].mean():.2f}, median={df[col].median():.2f}, std_dev={df[col].std():.2f}, skewness={df[col].skew():.2f}."
    ai_insight = get_dynamic_insights_gemini(prompt, model)
    # Using custom HTML to change the text color to white
    st.markdown(f'<div style="color: white; padding: 10px;">Gemini Insight: {ai_insight}</div>', unsafe_allow_html=True)


def generate_line_chart_with_gemini_insights(df, columns, model):
    """
    Generate a line chart with dynamic insights from Gemini API.
    Args:
        df (pd.DataFrame): The dataset.
        columns (list): List of columns to visualize.
        model (GenerativeModel): The configured Gemini model.
    """
    st.write("A **Line Chart** is a type of chart that displays information as a series of data points connected by straight lines. "
                    "It is commonly used to visualize trends over time, showing changes in data across intervals or periods. "
                    "This visualization is particularly useful for identifying patterns, relationships, and fluctuations in datasets.")
    # Filter and clean data
    df_filtered = df[columns].dropna()
    if df_filtered.empty:
        st.warning("No valid data to plot after removing rows with missing values.")
        return
    
    # Create the line chart
    fig = px.line(
        df_filtered,
        x=df_filtered.index,
        y=columns,
        markers=True,
        title="Line Chart for Selected Columns",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Generate a more descriptive prompt
    prompt = (
        f"Analyze the trends, anomalies, and patterns in the following dataset columns: {columns}. "
        f"The data represents numeric values over an index (e.g., time or row indices). "
        f"Here are example values from the dataset:\n\n{df[columns].head(3).to_string(index=False)}\n\n"
        f"Focus on trends like increasing or decreasing behavior, anomalies like outliers or sudden spikes, "
        f"and patterns like cyclical or seasonal variations."
    )
    
    # Fetch insights from Gemini API
    ai_insight = get_dynamic_insights_gemini(prompt, model)
    st.markdown(f'<div style="color: white; padding: 10px;">Gemini Insight: {ai_insight}</div>', unsafe_allow_html=True)
    
    # Generate Gemini API insights
    prompt = f"Provide insights for a line chart visualization of the following columns and do not provide any negative thoughts about the dataset: {columns}. Comment on trends, anomalies, or any interesting patterns."
    ai_insight = get_dynamic_insights_gemini(prompt, model)
    st.markdown(f'<div style="color: white; padding: 10px;">Gemini Insight: {ai_insight}</div>', unsafe_allow_html=True)


def generate_box_plot_with_gemini_insights(df, col, model):
    """
    Generate a box plot with dynamic insights from Gemini API.
    Args:
        df (pd.DataFrame): The dataset.
        col (str): Column to visualize.
        model (GenerativeModel): The configured Gemini model.
    """
    st.write("A **Box Plot** is a graphical representation of data that displays its distribution through five key summary statistics: "
                    "minimum, first quartile (Q1), median, third quartile (Q3), and maximum. "
                    "It is useful for identifying outliers, understanding variability, and comparing distributions across different groups.")

    outliers = (
        (df[col] < (df[col].quantile(0.25) - 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25)))) |
        (df[col] > (df[col].quantile(0.75) + 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25))))
    ).sum()
    st.write(f"- Outliers detected: {outliers}")
    
    fig = px.box(df, y=col, title=f"Box Plot for {col}")
    st.plotly_chart(fig, use_container_width=True)
    
    # Generate Gemini API insights
    prompt = f"Provide insights for a box plot visualization of the column '{col}', highlighting outliers and the spread of the data."
    ai_insight = get_dynamic_insights_gemini(prompt, model)
    st.markdown(f'<div style="color: white; padding: 10px;">Gemini Insight: {ai_insight}</div>', unsafe_allow_html=True)


def generate_stacked_line_chart_with_gemini_insights(df, columns, model):
    """
    Generate a stacked line chart with dynamic insights from Gemini API.
    Args:
        df (pd.DataFrame): The dataset.
        columns (list): List of columns to visualize.
        model (GenerativeModel): The configured Gemini model.
    """
    st.write("A **Stacked Line Chart** is a variation of a line chart that displays multiple data series stacked on top of each other. "
                        "It shows the cumulative effect of the data series, making it easy to visualize the contribution of each series to the total. "
                        "This chart is particularly useful for understanding the composition and trends of data over time.")
    fig = px.area(
        df[columns],
        title="Stacked Line Chart",
        labels={"index": "Index"}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Generate Gemini API insights
    prompt = f"Provide insights for a stacked line chart visualization of the following columns: {columns}. Focus on overlapping trends and cumulative values."
    ai_insight = get_dynamic_insights_gemini(prompt, model)
    st.markdown(f'<div style="color: white; padding: 10px;">Gemini Insight: {ai_insight}</div>', unsafe_allow_html=True)


def generate_gauge_chart_with_gemini_insights(df, col, model):
    """
    Generate a gauge chart with dynamic insights from Gemini API.
    Args:
        df (pd.DataFrame): The dataset.
        col (str): Column to visualize.
        model (GenerativeModel): The configured Gemini model.
    """
    st.write("A **Gauge Chart** is a type of data visualization that represents a single data value within a range. "
                    "It resembles a speedometer and is often used to display performance metrics, progress towards a goal, or other key indicators. "
                    "This chart is particularly effective for quickly conveying whether a value falls within an acceptable range.")
    gauge_value = df[col].mean()
    st.write(f"- Average Value: {gauge_value:.2f}.")
    
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=gauge_value,
            title={'text': f"Gauge Chart for {col}"},
            gauge={
                'axis': {'range': [df[col].min(), df[col].max()]},
                'bar': {'color': "darkblue"}
            }
        )
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Generate Gemini API insights
    prompt = f"Provide insights for a gauge chart visualization of the column '{col}' with an average value of {gauge_value:.2f}."
    ai_insight = get_dynamic_insights_gemini(prompt, model)
    st.markdown(f'<div style="color: white; padding: 10px;">Gemini Insight: {ai_insight}</div>', unsafe_allow_html=True)

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
