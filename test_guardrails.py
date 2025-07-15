import streamlit as st
import requests
import dotenv
import os
import litellm

# --- Load environment variables ---
dotenv.load_dotenv()
together_api_key = os.getenv("TOGETHER_API_KEY")
if not together_api_key:
    st.error("TOGETHER_API_KEY not found. Please check your .env file.")
    st.stop()

model = "together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1"

# --- Set model alias ---
litellm.model_alias_map = {
    "together-qwen": 
    {
        "model_name": model,
        "litellm_provider": "together_ai",
        "api_key": together_api_key 
        }
}

# --- Streamlit page config ---
st.set_page_config(page_title="Guardrails Chatbot", layout="wide")
st.title("🛡️ Guardrails Chatbot (w/ Competitor Filter)")

# --- Chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Define TogetherAI call ---
def call_together_qwen(prompt, **kwargs):
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=kwargs.get("max_tokens", 1000),
        temperature=kwargs.get("temperature", 0.7)
    )
    return response.choices[0].message.content

# --- Guardrails call ---
def validate_guardrails(messages):
    payload = {
        "model": model,
        "messages": messages
    }
    response = requests.post(
        "http://localhost:8000/guards/competitor-guard/openai/v1/chat/completions",
        headers={"Authorization": "Bearer dummy"},
        json=payload
    )
    response.raise_for_status()
    return response.json()

# --- User input ---
user_input = st.chat_input("Say something...")
if user_input:
    # 1. Query Guard
    try:
        validate_guardrails([{"role": "user", "content": user_input.lower()}])
        st.session_state.messages.append({"role": "user", "content": user_input})
    except Exception as e:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": "🚫 Query blocked by Guardrails.\n" + str(e)})
        user_input = None

# 2. Model Response + Response Guard
    if user_input:
        try:
            response = call_together_qwen(user_input)
            # Validate model response
            validate_guardrails([
                {"role": "user", "content": user_input.lower()},
                {"role": "assistant", "content": response.lower()}
            ])
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"🚫 Blocked by Guardrails after model response.\n{str(e)}"
            })

# --- Display messages ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
