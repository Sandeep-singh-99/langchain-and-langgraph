from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
from langchain.tools import tool


@tool
def search(query: str) -> str:
    """Search for a query and return a short summary."""
    return f"Search results for: {query}"


agent = create_agent(
    model="google_genai:gemini-3.6-flash",
    tools=[search],
    middleware=[PIIMiddleware("email")],
)

"""
Some policies can’t live in a prompt—they need to be enforced deterministically regardless of what the model does. 
Guardrails intercept data as it flows through the agent loop, applying compliance rules or content policies before tool results reach the model’s context.
"""