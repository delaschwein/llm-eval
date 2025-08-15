import os
import json
import logging
import time
from dotenv import load_dotenv
import requests

load_dotenv()  # Loads .env if present
JUDGE_ONE_MODEL = os.getenv("JUDGE_ONE_MODEL", "openai/gpt-4o-mini")
JUDGE_TWO_MODEL = os.getenv("JUDGE_TWO_MODEL", "openai/gpt-4o-mini")
ARBITRATION_MODEL = os.getenv("ARBITRATION_MODEL", "openai/gpt-4o-mini")

logger = logging.getLogger(__name__)


def generate(
    prompt_text,
    system_text="You are a helpful assistant acting as an impartial judge.",
    model_name=JUDGE_ONE_MODEL,
    temperature=0.0,
    retries=10,  # Number of retries
):
    """
    Calls OpenAI's chat model to get Diplomacy orders using requests.
    The prompt_text should be structured JSON describing the game state.
    Retries the call up to `retries` times with a 5-second delay between attempts on failure.
    """
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    base_url = os.getenv("BASE_URL", "https://openrouter.ai/api/v1")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": prompt_text},
    ]

    """"response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "rating",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "verdict": {
                        "type": "boolean",
                        "description": "Is this an acceptable output?",
                    },
                    "justification": {
                        "type": "string",
                        "description": "A brief explanation of the verdict.",
                    },
                },
                "required": ["verdict", "justification"],
                "additionalProperties": False,
            },
        },
    },"""

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "top_k": 3,
        "max_tokens": 16000,
    }

    for attempt in range(1, retries + 1):
        try:
            response = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=120,
            )
            response.raise_for_status()

            llm_response = response.json()["choices"][0]["message"]["content"]
            logger.debug(f"LLM raw response: {llm_response}")
            # print(llm_response)
            return llm_response

        except requests.RequestException as e:
            logger.error(f"API request failed on attempt {attempt}: {e}")
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"Failed to parse response on attempt {attempt}: {e}")

        if attempt < retries:
            time.sleep(3)  # wait 5 seconds before retrying

    # Fallback after all retries have been exhausted
    # return {"journal_update": ["(Error calling LLM, fallback)"], "orders": []}
    return ""
