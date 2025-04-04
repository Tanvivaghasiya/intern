from langchain_core.tools import tool
@tool
def multiply(a:int,b:int)->int:
    """multiply A and b"""
    return a*b
print(multiply.name)
print(multiply.args)
print(multiply.description)