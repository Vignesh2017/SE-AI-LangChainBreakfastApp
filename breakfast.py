
import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv


load_dotenv()

DEFAULT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Configure your API key first."
        )

    # Initialize the model through LangChain
    llm = ChatOpenAI(
        model=DEFAULT_MODEL,
        temperature=0
    )

    # Specify the exact output format
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a healthy nutrition and breakfast expert. "
            "Suggest practical, nutritious breakfast options."
        ),
        (
            "human",
            "Suggest exactly 5 healthy breakfast ideas. "
            "Format the response as a numbered list from 1 to 5, "
            "with exactly one idea per line. "
            "Do not include a title, introduction, preamble, "
            "conclusion, or any text outside the five numbered lines."
        )
    ])

    # Build the LangChain chain and invoke it
    chain = prompt | llm
    response = chain.invoke({})

    # Print only the reply text, not the AIMessage object
    print(response.content)


if __name__ == "__main__":
    main()
