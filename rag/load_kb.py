import boto3
import os
import json

bedrock_kb_client = boto3.client('bedrock-agent-runtime', 'us-east-1')

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
)


def create_prompt(career_goals_summary):
    prompt = f"""Based on the user's career goals, recommend only 5 suitable Coursera courses
        Career goals: {career_goals_summary}
        """
    return prompt

def output_prompt(course_info):
    prompt= f"""Based on the recommended courses list given below, output the information into the format given:
        Recommended courses list: {course_info}

        <Output format>
        To achieve your goals, here are some recommended courses:
        1) Course Title: {{course_title}}
        Rating: {{rating}}
        Level: {{level}}
        URL: {{course_url}}

        2) Course Title: {{course_title}}
        Rating: {{rating}}
        Level: {{level}}
        URL: {{course_url}}

        ..."""
    return prompt

def call_kb_model(prompt):
    knowledgeBaseResponse  = bedrock_kb_client.retrieve_and_generate(
        input={'text': prompt},
        retrieveAndGenerateConfiguration={
            'knowledgeBaseConfiguration': {
                'knowledgeBaseId': 'OCSV8F0PEC',
                'modelArn': 'arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-haiku-20240307-v1:0'
            },
            'type': 'KNOWLEDGE_BASE'
        })
    return knowledgeBaseResponse

def call_mistral_model(prompt, max_tokens):
    if not prompt.strip():
        raise ValueError("Input prompt is empty. Please provide a valid prompt.")

    prompt_config = {
        "prompt": f"<s>[INST] {prompt} [/INST]",
        "max_tokens": max_tokens,
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

        if "outputs" in response_body and len(response_body["outputs"]) > 0:
            message_content = response_body["outputs"][0]["text"]
            return message_content
        else:
            raise ValueError("Unexpected response structure or empty content.")
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def summarize_career_goals(career_goals):
    prompt = f"Summarize the following career goals and extract key topics or skills:\n{career_goals}"
    summary = call_mistral_model(prompt, max_tokens=100)
    return summary

def extract_and_format_courses(response):
    # Extract the relevant details from the response
    text = response['citations'][0]['retrievedReferences'][0]['content']['text']
    print(text)
    return text


def getAnswers(career_goals_summary):
    prompt = create_prompt(career_goals_summary)
    response = call_kb_model(prompt)
    #print(response)
    formatted_output = extract_and_format_courses(response)
    new_prompt = output_prompt(formatted_output)
    output = call_mistral_model(new_prompt, max_tokens=4096)
    return output

def get_suggestions(career_goals):
    career_goals_summary = summarize_career_goals(career_goals)
    #overall_prompt = create_prompt(career_goals_summary)
    formatted_response = getAnswers(career_goals_summary)
    return formatted_response

# Example usage
#career_goals = "I want to become a data scientist and improve my skills in machine learning and data analysis."
#career_goals_summary = summarize_career_goals(career_goals)
#print("Career Goals Summary:", career_goals_summary)

#overall_prompt = create_prompt(career_goals_summary)
#formatted_response = getAnswers(career_goals_summary)
#print(formatted_response)