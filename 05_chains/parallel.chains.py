from pydantic import BaseModel, Field
from typing import Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="The sentiment of the feedback provided")

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template="Classify sentiment as positive or negative: {feedback}. Format: {format}",
    input_variables=["feedback"],
    partial_variables={"format": parser2.get_format_instructions()},
)

classifier_chain = prompt1 | model | parser2

prompt_pos = PromptTemplate(
    template="Write a positive reply to: {feedback}",
    input_variables=["feedback"],
)
prompt_neg = PromptTemplate(
    template="Write a negative reply to: {feedback}",
    input_variables=["feedback"],
)

parallel_chain = RunnableParallel(
    sentiment=classifier_chain,
    positive_reply=(prompt_pos | model | parser),
    negative_reply=(prompt_neg | model | parser),
)

result = parallel_chain.invoke({"feedback": "The phone is actually amazing"})
print(result)