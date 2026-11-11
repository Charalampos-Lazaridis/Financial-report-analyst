"""Checks on the model's numbers: inputs before a calculation, figures and answer text after the answer."""

import ast
import math
import re

from app.schemas import Answer, Calculation
from app.tools import calculate

NUMBER = r"\d+(?:\.\d+)?"  # a number like 11512 or 21.4
MIN_CHECKED = 1000  # smaller numbers are skipped, so 2, 100 and 365 stay allowed

def number_in_text(value: float, text: str) -> bool:
    """Check if a number appears in the text, written the way 10-Ks write numbers."""
    number = abs(value)
    candidates = {
        f"{number:,.0f}",  # 416161.0 -> "416,161"
        f"{number:,.1f}",  # 46.2     -> "46.2"
        f"{number:,.2f}",  # 6.08     -> "6.08"
    }
    if value < 0:
        # 10-Ks write negatives in parentheses: -4 -> "(4)"
        candidates = {f"({candidate})" for candidate in candidates}
    return any(candidate in text for candidate in candidates)


def calculator_values(calculations: list[Calculation]) -> list[float]:
    """Every value the calculator produced, including the parts of each expression:
    (11512 + 2763) / 86392 gives 0.1652, and also 14275 from the part in brackets."""
    values = [c.result for c in calculations]
    for c in calculations:
        for node in ast.walk(ast.parse(c.expression, mode="eval")):
            if isinstance(node, ast.BinOp):  # one operation, e.g. 11512 + 2763
                try:
                    values.append(calculate(ast.unparse(node)))
                except (ValueError, SyntaxError, ZeroDivisionError):
                    pass
    return values


def matches_calculation(value: float, results: list[float]) -> bool:
    """True if value is a calculator result, also when rounded (0.165 for 0.16524) or as a percent (16.52)."""
    text = f"{value}"
    decimals = len(text.split(".")[1]) if "." in text else 0  # 0.165 -> 3
    for result in results:
        for candidate in (result, result * 100):
            if math.isclose(value, candidate, rel_tol=0.001) or round(candidate, decimals) == value:
                return True
    return False


NUMBER_IN_TEXT = re.compile(r"(\d[\d,]*\.\d+)\s*%?|(\d[\d,]*)\s*%")  # decimals like 2.72, or percentages like 21%

def unbacked_numbers(answer: str, question: str, values: list[float]) -> list[str]:
    """Decimals and percentages in the answer text that match no listed figure or calculator result.
    Years and plain counts are skipped (too common), and so are numbers the question itself contains."""
    unbacked = []
    for match in NUMBER_IN_TEXT.finditer(answer):
        text = match.group(1) or match.group(2)
        if text in question:
            continue
        number = float(text.replace(",", ""))
        decimals = len(text.split(".")[1]) if "." in text else 0
        tolerance = 0.5 * 10 ** -decimals + 1e-9  # '2.72' matches anything from 2.715 to 2.725
        # a value may be stored as a fraction (0.1652 -> 16.52%), in millions ($1.5 billion), or negative
        candidates = [abs(v * scale) for v in values for scale in (1, 100, 1 / 1000)]
        if not any(abs(c - number) <= tolerance for c in candidates):
            unbacked.append(text)
    return unbacked


def fix_sources(result: Answer, chunks: list[dict], steps: list[str]) -> None:
    """Move a figure's citation when its number isn't in the cited excerpt but is in exactly one other.
    (A right number with a wrong citation; a number found in several excerpts is left for check_figures.)"""
    for figure in result.figures:
        if not 1 <= figure.source <= len(chunks) or number_in_text(figure.value, chunks[figure.source - 1]["text"]):
            continue
        found = [i for i, chunk in enumerate(chunks, start=1) if number_in_text(figure.value, chunk["text"])]
        if len(found) == 1:
            steps.append(f"Moved '{figure.name}' from excerpt [{figure.source}] to [{found[0]}], "
                         f"where {figure.value:,g} appears")
            figure.source = found[0]


def check_figures(result: Answer, chunks: list[dict], calculations: list[Calculation], question: str) -> list[str]:
    """Return a warning for every figure that doesn't match its excerpt or a calculator result."""
    results = calculator_values(calculations)
    warnings = []
    for figure in result.figures:
        if figure.source == 0:
            if not matches_calculation(figure.value, results):
                warnings.append(f"'{figure.name}' ({figure.value}) doesn't match any calculator result")
        elif not 1 <= figure.source <= len(chunks):
            warnings.append(f"'{figure.name}' cites excerpt [{figure.source}], which doesn't exist")
        elif (not number_in_text(figure.value, chunks[figure.source - 1]["text"])
              and not matches_calculation(figure.value, results)):
            warnings.append(f"'{figure.name}' ({figure.value}) not found in excerpt [{figure.source}]")
    for text in unbacked_numbers(result.answer, question, [f.value for f in result.figures] + results):
        warnings.append(f"'{text}' in the answer text matches no listed figure or calculator result")
    for c in calculations:
        tree = ast.parse(c.expression, mode="eval")
        if any(isinstance(node, ast.Constant) and node.value == 0 for node in ast.walk(tree)):
            warnings.append(f"Calculation {c.expression} uses 0 as an input: a missing value may have been filled in")
    return warnings

def unknown_inputs(expression: str, chunks: list[dict], calculations: list[Calculation]) -> list[str]:
    """Return the numbers in a calculation that aren't in the excerpts or earlier calculator results."""
    text = " ".join(chunk["text"] for chunk in chunks).replace(",", "")  # 11,512 -> 11512
    known = {float(number) for number in re.findall(NUMBER, text)}
    unknown = []
    for number in re.findall(NUMBER, expression):
        value = float(number)
        if value < MIN_CHECKED or value in known:
            continue
        if any(math.isclose(value, c.result, rel_tol=0.001) for c in calculations):
            continue  # it comes from an earlier calculation
        unknown.append(number)
    return unknown
