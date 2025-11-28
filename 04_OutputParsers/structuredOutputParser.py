from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

load_dotenv()

# ------------------------
# 1. Define Output Schema
# ------------------------
class BlackHoleFacts(BaseModel):
    fact1: str = Field(description="first fact about black hole")
    fact2: str = Field(description="second fact about black hole")
    fact3: str = Field(description="third fact about black hole")

parser = JsonOutputParser(pydantic_object=BlackHoleFacts)

# ------------------------
# 2. LLM Model
# ------------------------
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# ------------------------
# 3. Prompt
# ------------------------
template = PromptTemplate(
    template="""
    Give me 3 facts about {topic}.
    Return ONLY valid JSON matching this format:

    {format_instructions}
    """,
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# ------------------------
# 4. Chain
# ------------------------
chain = template | model | parser

# ------------------------
# 5. Invoke
# ------------------------
result = chain.invoke({"topic": "black holes"})
print(result)
