import os
import httpx

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

def get_gemini_response(prompt):
    try:
        # response = client.models.generate_content(
        #     # model="gemini-3.5-flash-lite",
        #     model="gemini-3.6-flash",
        #     contents=prompt
        # )
        # return response.text
        raise httpx.ReadTimeout("Simulated timeout")

    except httpx.TimeoutException:
        return f"Превышен таймаут клиента ({timeout_seconds} секунд)."

    except errors.ServerError as e:
        if e.code == 504:
            return "Превышено время ожидания ответа сервера (504)."
        return f"Ошибка сервера: {e}"

    except Exception as e:
        return f"Ошибка: {e}"


try:
    response = get_gemini_response("How are you doing?")
    print(response)
finally:
    client.close()
