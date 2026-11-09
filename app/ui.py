import os
import re

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/ask")

st.title("Financial Report Analyst")
st.caption(
    "Ask a factual question about these 10-K reports: Apple 2025; AMD, American Express, Boeing, PepsiCo, 3M, "
    "AES, CVS Health, Johnson & Johnson, Verizon 2022; Amcor, Best Buy 2023. Every number comes with its source."
)

question = st.text_input(
    "Your question",
    placeholder="What was AMD's net revenue in 2022?",
    max_chars=500,
)

if st.button("Ask"):
    with st.spinner("Searching the reports..."):
        try:
            response = requests.post(API_URL, json={"question": question}, timeout=(10, 120))
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

    st.subheader("Answer")
    st.write(result["answer"])
    st.write(f"Confidence: **{result['confidence']}**")

    if result["figures"]:
        st.subheader("Figures")
        st.table(result["figures"])

    for warning in result["warnings"]:
        st.warning(warning)

    # Show the excerpts the answer actually cites ([2] in the text, or a figure's source), not all 20.
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", result["answer"])}
    cited |= {figure["source"] for figure in result["figures"]}

    sources = result["sources"]
    if sources:
        rows = [
            {"excerpt": s["id"], "company": s["company"], "year": s["year"],
             "PDF page": "SEC XBRL" if s["page"] is None else str(s["page"] + 1)}
            for s in sources
        ]
        st.subheader("Sources")
        st.table([row for row in rows if row["excerpt"] in cited] or rows[:5])
        with st.expander(f"All {len(rows)} retrieved excerpts"):
            st.table(rows)

    steps = result.get("steps", [])  # .get: an older API version has no steps
    if steps:
        with st.expander("How I got this answer"):
            st.markdown("\n".join(f"- {s}" for s in steps))