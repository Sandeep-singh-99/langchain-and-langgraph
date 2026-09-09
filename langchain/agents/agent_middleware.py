from langchain.agents import create_agent
from deepagents.backends import StateBackend
from deepagents.middleware import FilesystemMiddleware, MemoryMiddleware, SkillsMiddleware, SummarizationMiddleware
from langchain.tools import tool
from langchain.agents.middleware import TodoListMiddleware
from deepagents.middleware.subagents import SubAgentMiddleware

# Tools
@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Search results for: {query}"

backend = StateBackend()

model="google_genai:gemini-2.5-flash",

# agent = create_agent(
#     model=model,
#     tools=[search],
#      middleware=[
#         FilesystemMiddleware(backend=backend),
#         SummarizationMiddleware(model=model, backend=backend),
#         MemoryMiddleware(backend=backend, sources=["./AGENTS.md"]),
#         SkillsMiddleware(backend=backend, sources=["./skills/"]),
#     ],
# )

agent = create_agent(
    model="google_genai:gemini-3.6-flash",
    tools=[search],
    middleware=[
        FilesystemMiddleware(backend=backend),
        TodoListMiddleware(),
        SubAgentMiddleware(
            backend=backend,
            subagents=[
                {
                    "name": "researcher",
                    "description": "Searches and returns a structured summary.",
                    "system_prompt": "Use the search tool to research the question and summarize key points.",
                    "tools": [search],
                    "model": "anthropic:claude-sonnet-4-6",
                    "middleware": [],
                }
            ],
        ),
    ],
)