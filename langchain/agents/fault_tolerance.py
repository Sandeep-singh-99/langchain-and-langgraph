from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware, ToolRetryMiddleware
from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv()


@tool
def search(query: str) -> str:
    """Search for a query and return a short summary."""
    return f"Search results for: {query}"


agent = create_agent(
    model="google_genai:gemini-3.6-flash",
    tools=[search],
    middleware=[
        ModelRetryMiddleware(max_retries=3),
        ToolRetryMiddleware(max_retries=2),
    ],
)

"""
Agents in production encounter failures that rarely appear in development: rate limits, model timeouts, transient API errors. 
Fault tolerance middleware handles these at the infrastructure level so your tools and business logic don’t need try/catch around every call.
"""
