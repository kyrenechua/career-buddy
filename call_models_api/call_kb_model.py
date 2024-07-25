import boto3

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