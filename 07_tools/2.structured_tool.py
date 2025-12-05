from langchain_core.tools import StructuredTool 
from pydantic import BaseModel, Field

class AddInput(BaseModel):
    a: int = Field(..., description="The first integer to add.")
    b: int = Field(..., description="The second integer to add.")

def add_numbers(a: int, b: int) -> int:
    """Adds two integers."""
    return a + b

add_tool = StructuredTool.from_function(
    func=add_numbers,
    name="add_numbers",
    description="Adds two integers together.",
    args_schema=AddInput
)

result = add_tool.invoke({"a": 10, "b": 15})
print("Addition Result:", result)
print("Tool Name:", add_tool.name)
print("Tool Description:", add_tool.description)
print("Tool arguments:", add_tool.args)