import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL","http://127.0.0.1:8000/ask")

st.title("Financial Report Analyst")
st.caption("Ask a factual question about Apple's 2025 10-K. Every number comes with its source.")

question = st.text_input(
    "Your question",
    placeholder="What were Apple's total net sales in fiscal 2025?",
    max_chars=500,
)

if st.button("Ask"):
    with st.spinner("Reading the 10-K..."):
        try:
            response = requests.post(API_URL, json={"question": question}, timeout=200)
        except requests.ConnectionError:
            st.error("Can't reach the API. Is uvicorn running?")
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