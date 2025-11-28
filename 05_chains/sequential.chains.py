from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

prompt1 = PromptTemplate(template="Write a detailed report on {topic}", input_variables=['topic'])

prompt2 = PromptTemplate(template="Write a 4 point summary on the following {text}", input_variables=['text'])

parser = StrOutputParser()

chains = prompt1 | model | parser | prompt2 | model | parser

result = chains.invoke({"topic": "nodejs vs python for backend development"})

print(result)