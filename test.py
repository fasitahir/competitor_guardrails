import dotenv
import os
import litellm
from typing import Optional

# --- Load environment variables ---
dotenv.load_dotenv()
together_api_key = os.getenv("TOGETHER_API_KEY")

def llm_detect_competitor(text: str, model_name: str = None) -> Optional[str]:
    system_prompt = (
        "Check whether the input text mentions any ticketing or booking companies other than Bookme.pk. "
        "Return exactly 'None' if no competitors are detected. "
        "If competitors are detected, return their names in lowercase, separated by commas, with no additional text. "
        "Exclude Bookme.pk and its variations (e.g., 'bookme', 'book me'). "
        "DO NOT include explanations, periods, or any other text. "
        "Examples: 'None', 'easytickets,tripadvisor'"
        "Example use cases: " \
        "If user mentions 'bookme', return 'None'. " \
        "If user mentions 'easytickets', return 'easytickets'. "
        "If user mentions 'bookme and tripadvisor', return 'tripadvisor'. " \
        "If user mentions 'bookme.pk and easytickets', return 'easytickets. "
        'User enters '' returns None. ' \
        "Do not return any other sentence or even word Just return the competitor names or 'None'. "

    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text}
    ]

    try:
        result = litellm.completion(
            model=model_name or "together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1",
            messages=messages,
            max_tokens=100,
            temperature=0.0
        )

        response = result.choices[0].message.content.strip().lower()

        # Check if response starts with "none" or is empty
        if response.startswith("none") or not response:
            return None

        # Explicitly filter out Bookme-related terms
        if any(term in response for term in {"bookme", "bookme.pk", "book me"}):
            return None

        # Split response into potential competitor names
        competitors = [name.strip() for name in response.split(",") if name.strip()]
        filtered_competitors = [
            name for name in competitors 
            if name and "bookme" not in name and name not in {"none", "no", "n/a", ""}
        ]

        # Return None if no valid competitors remain
        if not filtered_competitors:
            return None

        return ", ".join(filtered_competitors)

    except Exception as e:
        print(f"⚠️ LLM detection failed: {e}")
        return None

# Test cases
test_cases = [
    "Tell me about bookme",  # Should return None
    "I used EasyTickets for my flight",  # Should return "easytickets"
    "No competitors mentioned here",  # Should return None
    "Bookme and TripAdvisor are great",  # Should return "tripadvisor"
    "",  # Should return None
    "1+1"
]

for test in test_cases:
    result = llm_detect_competitor(test, model_name="together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1")
    print(f"Input: '{test}' -> Response: {result}")