from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser

from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

class BlackHoleFact(BaseModel):
    fact1: str = Field(..., description="first fact")
    fact2: str = Field(..., description="second fact")
    fact3: str = Field(..., description="third fact")

parser = PydanticOutputParser(pydantic_object=BlackHoleFact)

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

prompt = PromptTemplate(
    template="""
    Give me 3 facts about {topic}.

    Return JSON:
    {format_instructions}
    """,
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | model | parser

print(chain.invoke({"topic": "black holes"}))

