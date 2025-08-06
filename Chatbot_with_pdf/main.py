import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma


load_dotenv()

st.set_page_config(page_title="Chatbot with PDF", page_icon=":books:", layout="wide")
st.title("Chatbot with PDF")

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "store", "javascript.pdf")
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

if not os.path.exists(persistent_directory):
    st.write("Persistent directory does not exist. Please create it first.")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Read the document from the file
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Split the document from the file
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # Display the number of chunks created
    st.write("---- Document chunks information ----")
    st.write("Total number of chunks: ", {len(docs)})
    st.write("First chunk: ", {docs[0].page_content})

    # Create Embeddings
    st.write(" ---- Creating Embeddings ----")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    st.write("Embeddings created successfully.")

    # create a vector store
    st.write(" ---- Creating Vector Store ----")
    db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
    st.write("Vectors created successfully.")
else:
    st.write("Persistent directory already exists. Loading existing vector store.")
    db = Chroma(persist_directory=persistent_directory, embedding_function=GoogleGenerativeAIEmbeddings(model="gemini-1.5-flash"))