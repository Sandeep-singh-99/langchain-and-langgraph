from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

video_id = "VkC3XG7dlQ8"

try: 
    yt_api = YouTubeTranscriptApi()
    transcript_list = yt_api.fetch(video_id=video_id, languages=['en'])
    transcripts = " ".join(chunk.text for chunk in transcript_list)
except TranscriptsDisabled:
    transcripts = "No transcripts available for this video."

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
chunks = text_splitter.create_documents([transcripts])

print(len(chunks))

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma.from_documents(chunks, embedding=embedding_model)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

prompt = PromptTemplate(
    template="You are a helpful assistant. Answer only from the provided transcript context. If the context is insufficient, just answer i don't know.\n\nContext: {context}\n\nQuestion: {question}",
    input_variables=["context", "question"]
)

# question = "What is the main topic discussed in the video?"

question = "Can you summarize the key points covered in the video?"

retrieved_docs = retriever.invoke(question)
context = "\n".join([doc.page_content for doc in retrieved_docs])

final_prompt = prompt.invoke({"context": context, "question": question})

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

response = model.invoke(final_prompt)

print("Response:", response.content)