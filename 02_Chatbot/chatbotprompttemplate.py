from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

chat_prompt = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful {domain} assistant.'),
    ('human', 'Explain in simple terms, the concept of {topic}.'),
])

prompt = chat_prompt.invoke({
    "domain": "science",
    "topic": "quantum computing"
})

results = model.invoke(prompt)
print(results.content)