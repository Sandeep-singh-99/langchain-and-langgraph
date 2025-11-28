from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal
from langchain_classic.schema.runnable import RunnableBranch, RunnableLambda

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="The sentiment of the feedback provided")

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template="classify the sentiment of following positive or negative feedback {feedback} and provide the response in the following format: {response_format}",
    input_variables=['feedback'],
    partial_variables={"response_format": parser2.get_format_instructions()},
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(template="Write an appropriate response to the following positive feedback: {feedback}", input_variables=['feedback'])

prompt3 = PromptTemplate(template="Write an appropriate response to the following negative feedback: {feedback}", input_variables=['feedback'])


branch = RunnableBranch(
    (lambda x: x.sentiment == "positive", prompt2 | model | parser  ),
    (lambda x: x.sentiment == "negative", prompt3 | model | parser  ),
    RunnableLambda(lambda x: "No valid sentiment found.")
)

chain = classifier_chain | branch

result = chain.invoke({"feedback": "The phone is actually amazing"})
print(result)