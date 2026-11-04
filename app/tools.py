import ast
import operator


OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,  # a minus sign in front of a number: -4
}

CALCULATOR_TOOL = {
    "type": "function",
    "name": "calculate",
    "description": "Do arithmetic exactly. Use it for every calculation instead of working it out yourself.",
    "parameters": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Numbers and + - * / ( ) only, no commas or units, e.g. (23601 - 16434) / 16434",
            },
        },
        "required": ["expression"],
        "additionalProperties": False,
    },
    "strict": True,  # the API guarantees the arguments match this schema
}

def evaluate(node: ast.AST) -> float:
    """Work out a parsed expression, refusing anything that isn't basic arithmetic."""
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        return OPERATORS[type(node.op)](evaluate(node.left), evaluate(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPERATORS:
        return OPERATORS[type(node.op)](evaluate(node.operand))
    raise ValueError("only numbers and + - * / ( ) are allowed")

def calculate(expression: str) -> float:
    return float(evaluate(ast.parse(expression, mode="eval").body))