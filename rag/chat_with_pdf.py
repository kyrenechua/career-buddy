import json
import os
import shutil
import boto3
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    UnstructuredFileLoader,
)

file_path = "resume.pdf"

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


def call_claude_haiku(prompt):

    prompt_config = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }

    body = json.dumps(prompt_config)

    modelId = "anthropic.claude-3-haiku-20240307-v1:0"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body.get("content")[0].get("text")
    return results

def call_claude_mistral(prompt):

    prompt_config = {
        "prompt": f"<s>[INST] {prompt} [/INST]",
        "max_tokens": 4096,
        "temperature": 0.5,
        "top_p": 0.9,
        "top_k": 50
    }

    body = json.dumps(prompt_config)

    modelId = "mistral.mistral-small-2402-v1:0"
    accept = "application/json"
    contentType = "application/json"


    try:
        response = bedrock_runtime.invoke_model(
            body=body, modelId=modelId, accept=accept, contentType=contentType
        )
        response_body = json.loads(response.get("body").read())

        # Log the entire response for debugging
        print("Full Response Body:", response_body)

        # Handle the case where response_body might not have the expected structure
        if "outputs" in response_body and len(response_body["outputs"]) > 0:
            message_content = response_body["outputs"][0]["text"]
            return message_content
        else:
            raise ValueError("Unexpected response structure or empty content.")
    except Exception as e:
        print(f"An error occurred: {e}")
        return None




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

    return call_claude_haiku(prompt)

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

    return call_claude_mistral(prompt)


