from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal, Optional
from pydantic import BaseModel, Field

load_dotenv();


model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

class Review(BaseModel):
    key_theme: list[str] = Field(description="Write down 3 keys theme discussed in the review in a list")
    summary:str = Field(description="A brief summary of the review")
    setiment: Literal['positive', 'negative'] = Field(description="Return the setiment of the review either positive or negative")
    name: Optional[str] = Field(description="Write down the name of reviewer")

str_model = model.with_structured_output(Review, strict=True)

prompt = """
Google's Pixel phone have never been the most powerful handsets, with their Tensor chipsets falling behind rivals in benchmark. But surprisingly, the google pixel 10 series
might be even more compromised than pixel 9 series, at least when it comes to the GPU.

Reviewed by Sandeep Singh 
"""

results = str_model.invoke(prompt)
print(results)