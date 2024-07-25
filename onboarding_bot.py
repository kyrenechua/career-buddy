import base64
import io
import json
import os

import boto3
import streamlit as st

from index_handler import clear_local_index, delete_resume_file
from rag.chat_with_pdf import query_rag_with_bedrock, career_rag_with_bedrock, onboard_rag_with_bedrock


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

st.title("Nomura AI Powered Employee Onboarding Mentor")  # Title of the application
st.subheader("Created by Team F-Scholars")  # Subheader

# Initialize session state for resume file
if "resume_uploaded" not in st.session_state:
    st.session_state.resume_uploaded = False
    
# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

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
query = st.text_input("Enter your job role")

# Button to submit query
if st.button("Submit"):
    if st.session_state.resume_uploaded:
        with st.chat_message("assistant"):

            # Display the result of the query
            result = onboard_rag_with_bedrock(query)
            st.session_state.messages.append({"role": "assistant", "content": result})
            st.markdown(result)
            # st.session_state.query_result = result
        if prompt := st.chat_input("What is up?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                result = query_rag_with_bedrock(prompt)
                st.session_state.messages.append({"role": "assistant", "content": result})
                st.markdown(result)

    else:
        st.warning("No resume uploaded. Please upload a resume before submitting a query.")