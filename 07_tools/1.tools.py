from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two integers."""
    return a * b


result = multiply.invoke({"a": 6, "b": 7})
print("Multiplication Result:", result)
print("Tool Name:", multiply.name)
print("Tool Description:", multiply.description)
print("Tool arguments:", multiply.args)

print(multiply.args_schema.model_json_schema())

