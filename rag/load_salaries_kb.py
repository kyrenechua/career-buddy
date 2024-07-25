import boto3
import os
import json

from call_models_api.call_bedrock_runtime_models import call_mistral_model
from call_models_api.call_kb_model import call_kb_model

bedrock_kb_client = boto3.client('bedrock-agent-runtime', 'us-east-1')

def call_kb_model(kb_id, prompt):
    knowledgeBaseResponse  = bedrock_kb_client.retrieve_and_generate(
        input={'text': prompt},
        retrieveAndGenerateConfiguration={
            'knowledgeBaseConfiguration': {
                'knowledgeBaseId': kb_id,
                'modelArn': 'arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-haiku-20240307-v1:0'
            },
            'type': 'KNOWLEDGE_BASE'
        })
    return knowledgeBaseResponse

def create_prompt(current_monthly_salary, job_role, job_experience, job_location):
    prompt = f"""Based on the user's Job Role, job experience, and job location, generate the minimum, maximum, and median salary per month for the role, with their experience at their given work location.
                Look at their current monthly salary and determine if it is below or above the median salary for their role.
                The currency is in USD
                Follow the output format below.
        
        Job Role: {job_role}
        Job Experience: {job_experience}
        Job Location: {job_location}
        Current monthly salary: {current_monthly_salary}
        
        <Output format>
        Minimum Salary: USD {{min_salary}}\n
        Maximum Salary: USD {{max_salary}}\n
        Median Salary: USD {{median_salary}}\n

        Your current salary is {{below_above}} the median for your role.
        """
    return prompt

def output_prompt(text):
    prompt = f""" Look at the text given to you.
        Based on the text, determine if the person's current salary is below or above the median salary for their role.
        If it is below, suggest some ways they can negotiate their salary.
        If it is above, suggest some ways they can maintain their salary and tell them they are doing well.

        Text: {text}
    
    """
    return prompt

def get_salaries(current_monthly_salary, job_role, job_experience, job_location):
    salary_prompt = create_prompt(current_monthly_salary, job_role, job_experience, job_location)
    response = call_kb_model('CHYWIMBAIV', salary_prompt)
    output = response['output']['text']
    nego_prompt = output_prompt(output)
    nego_output = call_mistral_model(nego_prompt, 4096)
    return output + "\n" + nego_output

# Example usage
#current_monthly_salary = "50000"
#job_role = "Data Scientist"
#job_experience = "Entry level"
#job_location = "San Francisco"
#output = get_salaries(current_monthly_salary, job_role, job_experience, job_location)
#print(output)