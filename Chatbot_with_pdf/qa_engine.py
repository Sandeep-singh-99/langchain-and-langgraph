import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores.chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

load_dotenv()

def load_and_split(file_path: str):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt") or file_path.endswith(".md"):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError("Unsupported file type. Please upload a .pdf, .txt, or .md file.")

    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

def save_to_chroma(chunks):
    persist_directory = "chroma_store"
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    return vectordb

def get_qa_chain(vectordb, callbacks=[]):
    retriever = vectordb.as_retriever()
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", streaming=True, callbacks=callbacks)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
