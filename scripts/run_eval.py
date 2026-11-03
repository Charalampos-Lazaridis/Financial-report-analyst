import json
import math

from app.analyst import FALLBACK_ANSWER, AskResponse, ask, qdrant

def load_tests(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]

def passed(test: dict, result: AskResponse) -> bool:
    if result is FALLBACK_ANSWER:  # the API call failed: never counts as a pass
        return False
    if test["kind"] == "advice":
        return result.is_advice_request
    if test["kind"] == "not_in_report":
        return not result.figures and not result.is_advice_request

    expected = abs(test["value"])
    in_figures = any(math.isclose(abs(figure.value), expected, rel_tol=0.01) for figure in result.figures)
    in_answer = f"{expected:,}" in result.answer
    return in_figures or in_answer

tests = load_tests("data/eval_apple.jsonl")
passes = 0
for test in tests:
    result = ask(test["question"])
    ok = passed(test, result)
    passes += ok
    print(f"{'PASS' if ok else 'FAIL'}  {result.confidence:<6}  [{test['kind']}] {test['question']}")
    if not ok:
        print("        expected:", test.get("value", test["kind"]))
        print("        got:     ", result.answer)

print(f"\n{passes}/{len(tests)} passed")
qdrant.close()