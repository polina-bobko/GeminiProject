import os
import httpx

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception
)
from google import genai
from google.genai import types, errors
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

timeout_seconds = 10

client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(
        timeout=timeout_seconds * 1000,
        retry_options=types.HttpRetryOptions(attempts=1)
    )
)

def is_retryable(error):
    if isinstance(error, httpx.TimeoutException):
        return True

    if isinstance(error, errors.APIError):
        return error.code in (429, 500, 502, 503, 504)

    return False


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=10),
    retry=retry_if_exception(is_retryable),
    reraise=True
)
def get_gemini_response(prompt):
    print("Sending request...", flush=True)

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    return response.text


try:
    response = get_gemini_response("How are you doing?")
    print("Gemini response:")
    print(response)

except Exception as e:
    print(f"Ошибка после попыток запроса: {e}")

finally:
    client.close()