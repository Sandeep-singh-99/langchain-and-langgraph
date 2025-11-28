from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_community.chains import LLMChain, SequentialChain


llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write a short paragraph about {topic}."
)

chain = LLMChain(llm=llm, prompt=prompt)

result = chain.invoke({"topic": "Docker Compose"})
print(result["text"])
