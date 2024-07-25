import boto3
import json


bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
)


def call_claude_haiku(prompt, max_tokens):

    prompt_config = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
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