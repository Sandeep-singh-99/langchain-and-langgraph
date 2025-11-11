from dotenv import load_dotenv
import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="Chat Model Basic", page_icon=":robot:")
st.title("Chat Model Basic with Google")

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

result = model.invoke("What is the capital of India?")

st.write("Full Result")
st.write(result)
st.write("Result Content")
st.write(result.content)