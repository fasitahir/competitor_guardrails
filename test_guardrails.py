import os
import dotenv
import requests
import litellm

# Load environment variables
dotenv.load_dotenv()

# Get Together.ai API key
together_api_key = os.getenv("TOGETHER_API_KEY")
if not together_api_key:
    raise ValueError("TOGETHER_API_KEY not found in .env file. Get it from https://www.together.ai")

model = "together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1"

# Set model alias map
litellm.model_alias_map = {
    "together-qwen": {
        "model_name": model,
        "litellm_provider": "together_ai",
        "api_key": together_api_key
    }
}

# Function to call Together Qwen using LiteLLM
def call_together_qwen(prompt, **kwargs):
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=kwargs.get("max_tokens", 1000),
        temperature=kwargs.get("temperature", 0.7)
    )
    return response.choices[0].message.content

# Main execution
prompt = "Tell me about a platform where i can book tickets"

try:
    #Step 1: Send USER QUERY to Guardrails Server (query guard)
    query_payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt.lower()}
        ]
    }
    query_response = requests.post(
        "http://localhost:8000/guards/competitor-guard/openai/v1/chat/completions",
        headers={"Authorization": "Bearer dummy"},
        json=query_payload
    )
    query_response.raise_for_status()
    query_result = query_response.json()
    print("Query Guard Passed")

    # Step 2: Call Model
    qwen_response = call_together_qwen(prompt, max_tokens=1000)
    print("Qwen Response:", qwen_response)

    #Step 3: Send MODEL RESPONSE to Guardrails Server (response guard)
    response_payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt.lower()},
            {"role": "assistant", "content": qwen_response.lower()}
        ]
    }
    response = requests.post(
        "http://localhost:8000/guards/competitor-guard/openai/v1/chat/completions",
        headers={"Authorization": "Bearer dummy"},
        json=response_payload
    )
    response.raise_for_status()
    response_data = response.json()
    print("Guardrails Server Response:", response_data)

except requests.exceptions.RequestException as req_err:
    print(f"❌ Request Error: {req_err}")
except Exception as e:
    print(f"❌ Blocked by Guard: {str(e)}")
