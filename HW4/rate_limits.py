import time
import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("API key not found")

client = genai.Client(api_key=api_key)

last_request_time = 0
request_interval = 10  # Не чаще 1 запроса в n секунд

def get_gemini_response(prompt):
    global last_request_time

    elapsed = time.monotonic() - last_request_time

    if elapsed < request_interval:
        wait_time = request_interval - elapsed
        print(f"Waiting {wait_time:.2f} seconds...")
        time.sleep(wait_time)

    last_request_time = time.monotonic()

    print(f"Sending request at {time.strftime('%H:%M:%S')}")

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    return response.text

try:
    # response = get_gemini_response("How are you doing?")
    # print(response)
    for i in range(3):
        print(f"Request {i + 1}")
        print(get_gemini_response("Hello!"))
finally:
    client.close()