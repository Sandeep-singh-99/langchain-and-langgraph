from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, List

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

class Review(TypedDict):
    key_themes: Annotated[Optional[List[str]], "must write down all the important concepts discussed in the review in a list"]
    summary: Annotated[str, "must write down a brief summary of the review"]
    sentiment: Annotated[str, "must written a sentiment of the review, either postive or negative"]
    props: Annotated[Optional[list[str]], "write down all the props inside a list"]
    cons: Annotated[Optional[list[str]], "write down all the cons inside a list"]

# Structured output prompt

structured_model = model.with_structured_output(Review)

prompt = """
Google's Pixel phone have never been the most powerful handsets, with their Tensor chipsets falling behind rivals in benchmark. But surprisingly, the google pixel 10 series
might be even more compromised than pixel 9 series, at least when it comes to the GPU. 
"""

results = structured_model.invoke(prompt)
print(results)