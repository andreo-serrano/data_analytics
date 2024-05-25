import os
from pathlib import Path

import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv

from config import model
from src.utils import extract_csv_content, save_uploaded_file

# Optional: Consider using a more secure secret management solution for the API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def main():
    """
    The main function of the Streamlit app.
    """

    st.title("Discutez avec votre fichier csv")
    uploaded_file = st.file_uploader(
        "telechargez votre fichier csv", type="csv")
    save_directory = "data"  # You can change this to your desired directory

    if uploaded_file is not None:
        # Ensure the data directory exists
        # Create the directory if it doesn't exist, ignoring errors if it already exists
        os.makedirs(save_directory, exist_ok=True)
        saved_file_path = save_uploaded_file(uploaded_file, save_directory)
        chat_session = model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": extract_csv_content(saved_file_path)
                }
            ]
        )
        user_question = st.text_input("Posez des questions")
        if user_question:

            response = chat_session.send_message(user_question)
            st.write(response.text)


if __name__ == "__main__":
    main()
