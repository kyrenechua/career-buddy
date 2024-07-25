import boto3
import os
import json

bedrock_kb_client = boto3.client('bedrock-agent-runtime', 'us-east-1')

def create_prompt(job_role, job_experience, job_location):
    prompt = f"""Based on the user's Job Role, job experience, and job location, generate the minimum, maximum, and median salary per year for the role, with their experience at their given work location.
                Give some suggestions for the user to negotiate their salary.
        Job Role: {job_role}
        Job Experience: {job_experience}
        Job Location: {job_location}
        """
    return prompt

def output_prompt(job_role, job_experience, job_location):
    prompt= f"""Based on your job role, job experience and job location, these are the salary ranges:
        Job Role: {job_role}
        Job Experience: {job_experience}
        Job Location: {job_location}

        <Output format>
        The minimum salary for the role as a {job_experience} at {job_location} is: {{min_salary}}
        The maximum salary for the role is: {{max_salary}}
        The median salary for the role is: {{median_salary}}

        Here are some suggestions to negotiate your salary:
        1) Suggestion 1
        2) Suggestion 2
        3) Suggestion 3

        ...

        """
    return prompt

def call_kb_model(prompt):
    knowledgeBaseResponse  = bedrock_kb_client.retrieve_and_generate(
        input={'text': prompt},
        retrieveAndGenerateConfiguration={
            'knowledgeBaseConfiguration': {
                'knowledgeBaseId': 'CHYWIMBAIV',
                'modelArn': 'arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-haiku-20240307-v1:0'
            },
            'type': 'KNOWLEDGE_BASE'
        })
    return knowledgeBaseResponse



def get_salaries(job_role, job_experience, job_location):
    prompt = create_prompt(job_role, job_experience, job_location)
    response = call_kb_model(prompt)
    output = response['output']['text']
    return output

# Example usage
#job_role = "Data Scientist"
#job_experience = "Entry level"
#job_location = "San Francisco"
#output = getAnswers(job_role, job_experience, job_location)
#print(output)