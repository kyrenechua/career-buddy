import json
import os
import boto3

from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    UnstructuredFileLoader,
)

from call_models_api.call_bedrock_runtime_models import call_claude_haiku, call_mistral_model



REGION = "us-east-1"

# Setup bedrock
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
)


def chunk_doc_to_text(doc_loc: str):
    # Check if file exists
    if not os.path.exists(doc_loc):
        print(f"File not found: {doc_loc}")
        return None
    
    print("chunking doc to text!")
    try:
        loader = UnstructuredFileLoader(doc_loc)
        docs = loader.load()
        print("docs loaded!")
    except FileNotFoundError:
        print(f"File not found: {doc_loc}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        texts = text_splitter.split_documents(docs)
        if not texts:
            print("No texts were split.")
            return None
        print("Texts split successfully.")
    except Exception as e:
        print(f"An error occurred while splitting the texts: {e}")
        return None

    return texts



def query_rag_with_bedrock(query):

    embeddings = BedrockEmbeddings(
        client=bedrock_runtime,
        model_id="amazon.titan-embed-text-v2:0",
    )

    pdf_loc = "./pdf_files/resume.pdf"

    if os.path.exists("local_index"):
        local_vector_store = FAISS.load_local("local_index", embeddings,allow_dangerous_deserialization=True)
    else:
        texts = chunk_doc_to_text(pdf_loc)
        local_vector_store = FAISS.from_documents(
            texts, embeddings
        )
        local_vector_store.save_local("local_index")

    docs = local_vector_store.similarity_search(query)
    context = ""

    for doc in docs:
        context += doc.page_content

    prompt = f"""Use the following pieces of context to answer the question at the end.

    {context}

    Question: {query}
    Answer:"""

    return call_claude_haiku(prompt, max_tokens=1000)

def career_rag_with_bedrock(query):
    embeddings = BedrockEmbeddings(
        client=bedrock_runtime,
        model_id="amazon.titan-embed-text-v2:0",
    )

    pdf_loc = "./pdf_files/resume.pdf"

    if os.path.exists("local_index"):
        local_vector_store = FAISS.load_local("local_index", embeddings,allow_dangerous_deserialization=True)
    else:
        texts = chunk_doc_to_text(pdf_loc)
        local_vector_store = FAISS.from_documents(
            texts, embeddings
        )
        local_vector_store.save_local("local_index")

    docs = local_vector_store.similarity_search(query)
    context = ""

    for doc in docs:
        context += doc.page_content

    prompt = f"""You are a mentor for the person in the resume.
                Use the resume to provide career advice to the person, based on his or her career goals in the query.
                The career advice should include the following:
                - The person's strengths
                - The person's weaknesses
                - Hard skills he or she can learn or improve
                - Soft skills he or she can learn or improve
                - The person's potential career trajectory
                - Other career paths he or she can consider
                - The person's expected salary range
                - Any other advice you think is relevant and useful

    Hello! After looking at your resume and career goals, here are some advice I have for you!

    <career advice?
    
    I wish you all the best in your career! You can do it!
"""

    return call_mistral_model(prompt, max_tokens=4096)



def onboard_rag_with_bedrock(query):
    embeddings = BedrockEmbeddings(
        client=bedrock_runtime,
        model_id="amazon.titan-embed-text-v2:0",
    )

    pdf_loc = "./pdf_files/resume.pdf"

    if os.path.exists("local_index"):
        local_vector_store = FAISS.load_local("local_index", embeddings,allow_dangerous_deserialization=True)
    else:
        texts = chunk_doc_to_text(pdf_loc)
        local_vector_store = FAISS.from_documents(
            texts, embeddings
        )
        local_vector_store.save_local("local_index")

    docs = local_vector_store.similarity_search(query)
    context = ""

    for doc in docs:
        context += doc.page_content

    prompt = f"""You are a Human Resources representative that is in charge of onboarding a new employee into the company, Nomura, an international bank.
                The employee you have to onboard is the one stated in the resume.
                The employee is currently joining Nomura as a {query}.


                From the resume,  classify if the employee into either a fresh graduate or an experienced employee based on the methodology below.
                <Methodology>:
                    - If you see that the employee has only been an intern, classify him as a fresh graduate.
                    - If you see that the employee has had full time job experience, classify him as a an experienced employee. 
                Note:  that past internships are not considered experience.
                Note: you must classify the employee into only one category and this should be reflected in your output.

                Change the context of the onboarding accordingly:
                - For an experienced employee, they need not go through onboarding for technical skills, and would focus more on Nomura-specific standard operating procedures.
                - For a fresh graduate, you must inform them that they will go through technical skill training, as they are not proficient in their technical skills.
                    The technical skill training must be done in addition to Nomura-specific standard operating procedures.
                    This fresh hire will also be attached to a mentor in the company to guide him.

                Guide the employee through the onboarding procedures that he/she ha to go through.
                <Format>:
                Hi John, we are glad to have you join our team!
                I'm your Human Resources representative and I will be guiding you through the onboarding process.
                Based on your resume, you are a ...
    
"""

    return call_mistral_model(prompt, max_tokens=4096)

def chat_rag_with_bedrock(query):

    embeddings = BedrockEmbeddings(
        client=bedrock_runtime,
        model_id="amazon.titan-embed-text-v2:0",
    )

    pdf_loc = "./pdf_files/resume.pdf"

    if os.path.exists("local_index"):
        local_vector_store = FAISS.load_local("local_index", embeddings,allow_dangerous_deserialization=True)
    else:
        texts = chunk_doc_to_text(pdf_loc)
        local_vector_store = FAISS.from_documents(
            texts, embeddings
        )
        local_vector_store.save_local("local_index")

    docs = local_vector_store.similarity_search(query)
    context = ""

    for doc in docs:
        context += doc.page_content

    prompt = f"""Use the following pieces of context to answer the question at the end. If the answer cannot be found in the context, then just answer the question.

    {context}

    Question: {query}
    Answer:"""

    return call_claude_haiku(prompt, max_tokens=1000)


