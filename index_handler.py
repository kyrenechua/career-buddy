import os
import shutil


# Function to delete the local index directory
def clear_local_index():
    if os.path.exists("local_index"):
        shutil.rmtree("local_index")
        print("Local index directory cleared.")

# Function to delete the resume file
def delete_resume_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print("Resume file deleted.")
    else:
        print("No resume file to delete.")