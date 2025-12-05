from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

@tool
def add(a: int, b: int) -> int:
    """Adds two integers."""
    return a + b

llm_with_tool = model.bind_tools([add])

results = llm_with_tool.invoke("can you add 10 and 15 for me?")

print("LLM Result:", results)