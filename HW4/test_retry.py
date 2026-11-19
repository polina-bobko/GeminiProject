from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

# Счётчик попыток
attempt = 0


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    retry=retry_if_exception_type(ConnectionError),
    reraise=True,
)
def get_gemini_response(prompt):
    global attempt

    attempt += 1
    print(f"Attempt {attempt}: Sending request...")

    # Первые две попытки завершаются ошибкой
    if attempt <= 3:
        print("Error: 503 UNAVAILABLE")
        raise ConnectionError("503 UNAVAILABLE")

    # Третья попытка успешна
    print("Success!")
    return f"Gemini response: Hello! Your prompt was: {prompt}"


try:
    response = get_gemini_response("Hello!")
    print(response)

except Exception as e:
    print(f"Final error: {e}")
