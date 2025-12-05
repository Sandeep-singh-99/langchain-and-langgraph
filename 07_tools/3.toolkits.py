from langchain_core.tools import tool

@tool
def add(a: int, b: int) -> int:
    """Adds two integers."""
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """Subtracts the second integer from the first."""
    return a - b

@tool
def divide(a: int, b: int) -> float:
    """Divides the first integer by the second."""
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b

class MathToolkit:
    def get_tools(self):
        return [add, subtract, divide]

toolkit = MathToolkit()
tools = toolkit.get_tools()

for tl in tools:
    result = tl.invoke({"a": 20, "b": 5})
    print(f"{tl.name} Result:", result)
    print("Tool Name:", tl.name)
    print("Tool Description:", tl.description)
    print("Tool arguments:", tl.args)
    print(tl.args_schema.model_json_schema())