import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY не найден")

# Читаем текстовый файл
file_path = Path(__file__).with_name("knowledge.txt")
document = file_path.read_text(encoding="utf-8")

# Подключаем Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

# Создаём шаблон промпта
prompt = ChatPromptTemplate.from_template(
    """
    Ответь на вопрос только на основе документа.

    Если информации недостаточно, ответь:
    "В документе нет такой информации".

    Документ:
    {context}

    Вопрос:
    {question}

    Ответ на русском языке:
    """
)

# Создаём цепочку
chain = prompt | llm | StrOutputParser()

print("=== Django Document Q&A ===")

while True:
    question = input("\nВведите вопрос (exit для выхода): ")

    if question.lower() == "exit":
        break

    try:
        answer = chain.invoke({
            "context": document,
            "question": question
        })
        print(f"\nОтвет: {answer}")
    except Exception as e:
        print(f"Ошибка: {e}")