import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY не найден в .env")


def load_page(url):
    response = requests.get(
        url,
        timeout=20,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for element in soup(
        ["script", "style", "nav", "footer", "header"]
    ):
        element.decompose()

    text = soup.get_text(separator=" ", strip=True)

    if not text:
        raise ValueError("На странице не найден текст")

    return text[:12000]


llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

prompt = ChatPromptTemplate.from_template(
    """
    Ты помощник, который кратко пересказывает статьи.

    Прочитай текст и составь:
    1. Краткое содержание (3-5 предложений).
    2. Три главные идеи.
    3. Основной вывод.

    Отвечай на русском языке.
    Используй только информацию из текста.

    Текст:
    {text}
    """
)

chain = prompt | llm | StrOutputParser()


if __name__ == "__main__":
    url = "https://www.python.org/about/"

    try:
        text = load_page(url)
        print(f"Загружено {len(text)} символов")

        summary = chain.invoke({"text": text})

        print("\n=== SUMMARY ===")
        print(summary)

    except Exception as e:
        print(f"Ошибка: {e}")