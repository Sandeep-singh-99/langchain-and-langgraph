from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

class Review(TypedDict):
    summary: str
    sentiment: str

# Structured output prompt

structured_model = model.with_structured_output(Review)

prompt = """
This hardware is great, but the sotfware feels kind of bloated. So many boilerplate apps and my phone keeps hanging when i play pubg.
"""

results = structured_model.invoke(prompt)
print(results)