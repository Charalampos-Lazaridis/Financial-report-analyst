import os
import re
import time
from pathlib import Path

import requests
import streamlit as st

st.set_page_config(page_title="Financial Report Analyst", page_icon="📊", layout="centered")
# The look: background gradient, layout, cards (colours and font are in .streamlit/config.toml)
st.markdown(f"<style>{(Path(__file__).parent / 'style.css').read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True)

BADGE_COLORS = {"high": "green", "medium": "orange", "low": "red"}

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/ask")
REPO_URL = "https://github.com/Charalampos-Lazaridis/Financial-report-analyst"

COMPANIES = {
    "FY2025": ["Apple"],
    "FY2022": ["AMD", "American Express", "Boeing", "PepsiCo", "3M", "AES",
               "CVS Health", "Johnson & Johnson", "Verizon"],
    "FY2023": ["Amcor", "Best Buy"],
}
# Questions that show one feature each (all tested 2026-11-11)
DEMO_QUESTIONS = {
    "What was AMD's net revenue in 2022?": "simple lookup with a page citation",
    "What is PepsiCo's FY2022 EBITDA margin?": "calculation with the calculator tool",
    "Does AMD have a healthy liquidity profile based on its quick ratio for FY22?": "ratio + conclusion",
    "What was Boeing's research and development expense in FY2019?": "older year, live SEC lookup (MCP)",
    "What production rate changes is Boeing forecasting for FY2023?": "text answer from the report",
    "Is Apple stock a good investment right now?": "investment advice is declined",
}


def use_demo_question(text: str) -> None:
    """Button callback: put the demo question in the box; the user then clicks Ask."""
    st.session_state["question"] = text


st.title("Financial Report Analyst")
st.caption("Ask a factual question about a company's 10-K annual report. Every number comes with its source.")

with st.sidebar:
    name = st.text_input(
        "Your name (optional)",
        max_chars=40,
        help="With a name, the analyst remembers your answers that pass every check "
             "and recalls related ones when you ask something similar later.",
    )

    st.header("About")
    st.markdown(
        "An AI agent that answers questions about public companies from their SEC 10-K reports, "
        "with every number cited to a page.\n\n"
        "- **Retrieval** over 12 reports (Qdrant)\n"
        "- **Tools:** a calculator and a live SEC data lookup (MCP)\n"
        "- **Checks** on every number; doubtful answers get low confidence\n"
        "- **Memory** of your checked answers (with a name)\n\n"
        "**34/48** correct on FinanceBench questions, checked by hand.\n\n"
        f"[Code and write-up on GitHub]({REPO_URL})"
    )

    st.header("Companies")
    for year, names in COMPANIES.items():
        st.markdown(f"**{year}:** {', '.join(names)}")

    st.header("Try a question")
    for text, shows in DEMO_QUESTIONS.items():
        st.button(text, help=f"Shows: {shows}", on_click=use_demo_question, args=(text,),
                  width="stretch")

question = st.text_input(
    "Your question",
    key="question",  # the demo buttons write into this box through st.session_state
    placeholder="What was AMD's net revenue in 2022?",
    max_chars=500,
)

if st.button("Ask", type="primary"):
    with st.spinner("Searching the reports..."):
        try:
            started = time.perf_counter()
            response = requests.post(API_URL, json={"question": question, "user": name or None},timeout=(10, 120))
            seconds = time.perf_counter() - started  # wall time as the user feels it, network included
        except requests.ConnectionError:
            st.error("Can't reach the API. Is uvicorn running?")
            st.stop()
        except requests.Timeout:
            st.error("The API took too long to answer. Please try again.")
            st.stop()

    if response.status_code == 422:
        st.warning("Please type a question.")
        st.stop()
    if response.status_code != 200:
        st.error(f"The API returned an error ({response.status_code}).")
        st.stop()

    result = response.json()

    with st.container(border=True):  # the answer card
        confidence = result["confidence"]
        st.badge(f"{confidence} confidence", color=BADGE_COLORS.get(confidence, "gray"))
        st.markdown(f"#### {result['answer']}" if len(result["answer"]) < 120 else result["answer"])

    # The "receipt": what this answer cost and how much work it took
    model_calls = sum(1 for s in result.get("steps", []) if s.startswith("Step "))  # one "Step n:" per model call
    calculations = len(result.get("calculations", []))
    receipt = [f"${result['cost_usd']:.4f}", f"{model_calls} model call{'s' * (model_calls != 1)}"]
    if calculations:
        receipt.append(f"{calculations} calculation{'s' * (calculations != 1)}")
    receipt.append(f"{seconds:.1f} s")
    st.caption(" · ".join(receipt))

    if result["figures"]:
        st.subheader("Figures")
        figure_rows = [
            {"figure": f["name"], "value": f"{f['value']:,g}", "unit": f["unit"], "period": f["period"],
             "source": "calculator" if f["source"] == 0 else f"[{f['source']}]"}
            for f in result["figures"]
        ]
        st.dataframe(figure_rows, hide_index=True, width="stretch")

    for warning in result["warnings"]:
        st.warning(warning)

    if name:  # memory is on only with a name
        if result.get("remembered"):  # .get: an older API version has no memory fields
            st.caption("✓ Saved to your memory: this answer passed every check.")
        else:
            st.caption("Not saved to your memory: only answers that pass every check are remembered.")
    memories = result.get("memories", [])
    if memories:
        with st.expander(f"From your earlier analyses ({len(memories)})"):
            for m in memories:
                st.markdown(f"**{m['date']}** · {m['question']}  \n{m['answer']}")

    # Show the excerpts the answer actually cites ([2] in the text, or a figure's source), not all 20.
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", result["answer"])}
    cited |= {figure["source"] for figure in result["figures"]}

    sources = result["sources"]
    if sources:
        rows = [
            {"excerpt": f"[{s['id']}]", "company": s["company"], "year": str(s["year"]),  # text: left-aligned
             "PDF page": "SEC XBRL" if s["page"] is None else str(s["page"] + 1)}
            for s in sources
        ]
        st.subheader("Sources")
        cited_rows = [row for row, s in zip(rows, sources) if s["id"] in cited]
        st.dataframe(cited_rows or rows[:5], hide_index=True, width="stretch")
        with st.expander(f"All {len(rows)} retrieved excerpts"):
            st.dataframe(rows, hide_index=True, width="stretch")

    steps = result.get("steps", [])  # .get: an older API version has no steps
    if steps:
        with st.expander("How I got this answer"):
            st.markdown("\n".join(f"- {s}" for s in steps))