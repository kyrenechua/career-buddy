import base64
import io
import json
import os

import boto3
import streamlit as st

from index_handler import clear_local_index, delete_resume_file
from rag.chat_with_pdf import query_rag_with_bedrock, file_path, career_rag_with_bedrock
from rag.load_kb import get_suggestions

REGION = "us-east-1"

# Define bedrock
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name=REGION,
)

# Define the file path for the resume
file_path = "pdf_files/resume.pdf"

# Initialize session state for the first run to clear local index and delete the file
if "initialized" not in st.session_state:
    print("initializing session state")
    clear_local_index()
    delete_resume_file(file_path)
    st.session_state.initialized = True

st.title("Growth Mentor")  # Title of the application
st.subheader("I am the Growth Mentor. How can I help you today?")  # Subheader

# Initialize session state for resume file
if "resume_uploaded" not in st.session_state:
    st.session_state.resume_uploaded = False

# Streamlit file uploader for resume (only pdfs for now)
user_resume = st.file_uploader("Upload your resume", type=["pdf"], key="resume_uploader")

# Handle file upload and maintain the session state
if user_resume is not None:
    with open(file_path, "wb") as f:
        f.write(user_resume.read())
    st.success("Resume has been uploaded successfully!")
    st.session_state.resume_uploaded = True
    st.session_state.resume_file = user_resume

# Check if resume is uploaded and display it
if st.session_state.resume_uploaded:
    # Button to delete the resume
    if st.button("Delete Resume"):
        delete_resume_file(file_path)
        clear_local_index()
        st.session_state.resume_uploaded = False
        st.session_state.resume_file = None
        st.success("Resume deleted successfully!")

# User input for query
query = st.text_input("Enter your query")

# Button to submit query
if st.button("Submit Query"):
    if st.session_state.resume_uploaded:
        st.write(f"Query: {query}")

        # Display the result of the query
        result = query_rag_with_bedrock(query)
        st.session_state.query_result = result
        st.write("Result:")
        st.write(result)
    else:
        st.warning("No resume uploaded. Please upload a resume before submitting a query.")

# Career Mentor feature
st.subheader("Career Mentor")  # Subheader for Career Mentor feature

# User input for career goals
career_goals = st.text_area("Enter your career goals")

# Button to get career suggestions
if st.button("Get Career Suggestions"):
    if st.session_state.resume_uploaded:
        st.write(f"Career Goals: {career_goals}")

        # Generate career progression suggestions
        career_suggestions = career_rag_with_bedrock(career_goals)
        st.session_state.career_suggestions = career_suggestions
        st.write("Career Suggestions:")
        st.write(career_suggestions)
    else:
        st.warning("No resume uploaded. Please upload a resume before getting career suggestions.")

st.subheader("Coursera Buddy") # Subheader for Coursera Buddy feature
if (st.button("Get Coursera Suggestions")):
    st.write("Coursera Suggestions:")
    st.write(get_suggestions(career_goals))
