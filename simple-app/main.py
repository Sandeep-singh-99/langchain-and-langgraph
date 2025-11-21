from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
from dotenv import load_dotenv
import os
# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.title("Google Generative AI with LangChain")
question = st.text_input("Ask something...")

# LangChain setup
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user's query."),
    ("user", "Question: {question}")
])

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# Output
if question:
    with st.spinner("Thinking..."):
        response = chain.invoke({"question": question})
    st.success("Response:")
    st.write(response)

    