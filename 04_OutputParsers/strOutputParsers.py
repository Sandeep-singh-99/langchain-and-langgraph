from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# 1st prompt
template = PromptTemplate(template="Write a detailed report on {topic}", input_variables=['topic'])

# prompt1 = template.invoke({"topic": "English Premier league 2023/2024"})

# result1 = model.invoke(prompt1).content

# 2st prompt

template2 = PromptTemplate(template="Writea 4 point summary on the following {text}", input_variables=['text'])

# prompt2 = template2.invoke({'text':str(result1)})

# result = model.invoke(prompt2)

parser = StrOutputParser()

chain = template | model | parser | template2 | model | parser

result = chain.invoke({"topic": "English Premier league 2023/2024"})

print(result)