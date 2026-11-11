"""The agent loop: the prompt, the model calls, and the tools (calculator, SEC lookup through MCP)."""

import asyncio
import json
from functools import cache

from fastmcp import Client
from openai import OpenAI

from app import config
from app.checks import unknown_inputs
from app.retrieval import format_excerpts, format_musd
from app.schemas import Answer, Calculation
from app.sec_mcp_server import mcp as sec_server
from app.tools import CALCULATOR_TOOL, calculate

MAX_STEPS = 4  # most model calls for one question

INSTRUCTIONS = (
    "You are a financial analyst and thus tend to explain your answers. Answer the question using only the numbered excerpts from 10-K reports. "
    "Each excerpt is labelled with its company, fiscal year and page.\n"
    "- The last excerpt may be key figures from the company's SEC XBRL data, in millions of USD, for the "
    "report's fiscal year and the year before. Prefer its values for those figures and cite it like any excerpt.\n"
    "- If a number you need isn't in the excerpts (for example an earlier year, or a figure missing from the "
    "key figures), look it up with get_sec_fact, then cite the excerpt number it returns.\n"
    "- Start with a direct conclusion (yes/no, the number, or the main drivers), then support it in 2-4 sentences "
    "with the specific numbers, names and reasons from the excerpts.\n"
    "- If the question needs a calculation (ratio, margin, turnover, growth, year-over-year change), "
    "find the inputs in the excerpts, call the calculate tool, and show the formula and the inputs. "
    "Never do arithmetic yourself. In figures, give a calculated value with source 0 and copy the tool's "
    "result exactly; you may round it in the answer text.\n"
    "- Some questions say what to answer in a special case (for example, if a metric doesn't apply). "
    "Use that special answer only when the excerpts clearly show that case. "
    "If the inputs for the calculation are in the excerpts, calculate instead.\n"
    "- Say that the reports don't contain the answer only if the excerpts have nothing relevant "
    "and get_sec_fact can't find the number. "
    "Never replace a missing input with a different number.\n"
    "- Use these definitions unless the question gives its own: "
    "inventory turnover = cost of sales / average inventory; "
    "quick ratio = (cash + short-term investments + receivables) / current liabilities; "
    "return on assets = net income / average total assets; "
    "EBITDA = operating income + depreciation and amortization; a net loss is a negative number.\n"
    "- Do each calculation once. When you have the results you need, write your answer.\n"
    "- For a trend or comparison question (improving, declining, larger, smaller), calculate the values "
    "first, then draw the conclusion from the results.\n"
    "- Answer only the years, figures and companies the question asks for. Don't add extra years or metrics.\n"
    "Cite the excerpts you use in the answer, like [2]. "
    "List every number you use in figures, with source set to the number of the excerpt it comes from. "
    "Copy each value exactly as the report writes it (e.g. in millions if the table is in millions). "
    "Set is_advice_request to true only if the user asks for a recommendation or opinion on what to do "
    "with an investment (buy, sell, hold, or whether a stock is a good investment). "
    "Questions that ask for facts, metrics or calculations are not advice, "
    "even if they mention investors or an analyst's perspective."
)

client = OpenAI(api_key=config.OPENAI_API_KEY, max_retries=3, timeout=45)

# --- SEC lookups through MCP: our agent is a client of app/sec_mcp_server.py ---
# Client(sec_server) speaks MCP to the server in memory. Client("https://.../mcp") would speak the same
# protocol over HTTP to a server elsewhere: only that argument changes.

async def _list_sec_tools() -> list:
    async with Client(sec_server) as mcp_client:
        return await mcp_client.list_tools()


async def _call_sec_tool(name: str, arguments: dict):
    async with Client(sec_server) as mcp_client:
        return await mcp_client.call_tool(name, arguments, raise_on_error=False)


@cache
def sec_tools() -> list[dict]:
    """Ask the MCP server which tools it has (tool discovery), once, in OpenAI's tool format."""
    return [
        {"type": "function", "name": tool.name, "description": tool.description,
         "parameters": tool.input_schema, "strict": False}
        for tool in asyncio.run(_list_sec_tools())
    ]


def run_sec_tool(call, chunks: list[dict], steps: list[str]) -> str:
    """Send the model's SEC lookup to the MCP server. The result becomes a new excerpt the model can cite."""
    arguments = json.loads(call.arguments)
    result = asyncio.run(_call_sec_tool(call.name, arguments))
    text = result.content[0].text if result.content else "no result"
    if result.is_error:
        steps.append(f"SEC lookup {arguments} failed: {text}")
        return f"Error: {text}"  # the model sees the reason, e.g. the list of valid names, and can retry

    fact = result.data
    chunks.append({
        "doc_name": "", "company": fact["company"], "year": fact["fiscal_year"], "page": None,
        "chunk": f"sec-{fact['tag']}-{fact['fiscal_year']}", "score": 0.0,
        "text": (f"SEC XBRL lookup: {fact['figure']} [XBRL tag {fact['tag']}], fiscal year {fact['fiscal_year']} "
                 f"(period ending {fact['period_end']}): {format_musd(fact['value'])} {fact['unit']}"),
    })
    steps.append(f"SEC lookup via MCP: {fact['company']} {fact['figure']} FY{fact['fiscal_year']} = "
                 f"{format_musd(fact['value'])} {fact['unit']}, added as excerpt [{len(chunks)}]")
    return f"Excerpt [{len(chunks)}]: {chunks[-1]['text']}"


def run_tool(call, chunks: list[dict], calculations: list[Calculation], steps: list[str]) -> str:
    """Run one tool the model asked for, note it in steps, and return the text the model gets back."""
    if call.name != "calculate":
        return run_sec_tool(call, chunks, steps)
    expression = json.loads(call.arguments)["expression"]
    unknown = unknown_inputs(expression, chunks, calculations)
    if unknown:
        print(f"rejected: {expression} ({', '.join(unknown)} not found)")
        steps.append(f"Rejected `{expression}`: {', '.join(unknown)} isn't in the excerpts, so the model must recheck")
        return (f"Error: {', '.join(unknown)} isn't in the excerpts or your earlier results. "
                "Check the number and use the value exactly as the report writes it.")
    try:
        value = calculate(expression)
    except (ValueError, SyntaxError, ZeroDivisionError) as error:
        steps.append(f"Calculator error on `{expression}`: {error}")
        return f"Error: {error}"  # the model sees this and can fix its expression
    calculations.append(Calculation(expression=expression, result=value))
    print(f"calculate: {expression} = {value}")
    steps.append(f"Calculated `{expression}` = {round(value, 4):,}")
    return str(value)


def run_agent(question: str, chunks: list[dict], steps: list[str], background: str = ""):
    """Call the model, running the tools it asks for, until it answers or MAX_STEPS runs out.
    background: recalled memories, put before the excerpts (no excerpt numbers, so they can't be cited)."""
    content = f"Excerpts:\n\n{format_excerpts(chunks)}\n\nQuestion: {question}"
    if background:
        content = f"{background}\n\n{content}"
    messages = [{"role": "user", "content": content}]
    calculations = []
    input_tokens = output_tokens = 0
    seen = set()        # (tool, arguments) pairs already run
    answer_now = False  # set when a step only repeated earlier calls

    for step in range(MAX_STEPS):
        last_step = step == MAX_STEPS - 1
        response = client.responses.parse(
            model=config.OPENAI_MODEL,
            instructions=INSTRUCTIONS,
            input=messages,
            tools=[CALCULATOR_TOOL, *sec_tools()],
            tool_choice="none" if last_step or answer_now else "auto",  # tools off: must answer now
            text_format=Answer,
        )

        input_tokens += response.usage.input_tokens
        output_tokens += response.usage.output_tokens

        tool_calls = [item for item in response.output if item.type == "function_call"]
        if not tool_calls:  # the model is done
            result = response.output_parsed
            if answer_now:  # the work was done; the model was only repeating itself
                steps.append(f"Step {step + 1}: the model repeated earlier tool calls, so tools were "
                             "switched off and it wrote its answer")
            elif last_step and result:  # forced with the work possibly unfinished
                print("Forced an answer on the last step")
                result.confidence = "low"
                steps.append(f"Step {step + 1}: last step, tools off, so the model had to answer now "
                             "(confidence set to low)")
            else:
                steps.append(f"Step {step + 1}: the model wrote its answer")
            return result, calculations, input_tokens, output_tokens

        steps.append(f"Step {step + 1}: the model asked for {len(tool_calls)} tool call(s)")
        # same tool + same arguments (written in one fixed form, so spacing doesn't matter) = a repeat
        keys = {(c.name, json.dumps(json.loads(c.arguments), sort_keys=True)) for c in tool_calls}
        if keys <= seen:  # nothing new this step: switch tools off for the next call
            answer_now = True
        seen |= keys

        for call in tool_calls:
            messages.append({"type": "function_call", "call_id": call.call_id,
                             "name": call.name, "arguments": call.arguments})
            messages.append({"type": "function_call_output", "call_id": call.call_id,
                             "output": run_tool(call, chunks, calculations, steps)})

    print(f"Agent stopped: no answer after {MAX_STEPS} steps")
    return None, calculations, input_tokens, output_tokens
