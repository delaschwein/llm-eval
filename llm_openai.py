import os
import json
import logging
import time
from dotenv import load_dotenv
import openai

load_dotenv()  # Loads .env if present
JUDGE_ONE_MODEL = os.getenv("JUDGE_ONE_MODEL", "gpt-3.5-turbo")
JUDGE_TWO_MODEL = os.getenv("JUDGE_TWO_MODEL", "gpt-3.5-turbo")
ARBITRATION_MODEL = os.getenv("ARBITRATION_MODEL", "gpt-3.5-turbo")

logger = logging.getLogger(__name__)


def generate(
    prompt_text,
    system_text="You are a helpful assistant acting as an impartial judge.",
    model_name=JUDGE_ONE_MODEL,
    temperature=0.0,
    retries=10,  # Number of retries
):
    api_key = os.getenv("OPENAI_API_KEY", "")
    client = openai.OpenAI(api_key=api_key)

    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": prompt_text},
    ]

    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=2048,
            )

            llm_response = response.choices[0].message.content
            logger.debug(f"LLM raw response: {llm_response}")
            return llm_response

        except Exception as e:
            logger.error(f"API request failed on attempt {attempt}: {e}")

        if attempt < retries:
            time.sleep(3)  # wait 3 seconds before retrying

    return ""
