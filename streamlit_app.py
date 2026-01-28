# streamlit_app.py

import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from groq import Groq
import logging

# Setup
logging.basicConfig(level=logging.INFO)
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# LLM Wrapper

def call_llm(prompt: str, model: str = "llama-3.1-8b-instant") -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an AI fund management analyst, GP assistant, and LP liaison."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

# Functional Modules

def generate_deal_memo(df: pd.DataFrame) -> str:
    prompt = f"Create a full investment deal memo based on: {df.head(3).to_dict()}"
    return call_llm(prompt)

def summarize_legal_doc(text: str) -> str:
    prompt = f"Summarize this fund legal document in plain English: {text}"
    return call_llm(prompt)

def generate_capital_call(df: pd.DataFrame) -> str:
    prompt = f"Write a capital call notice based on: {df.head(3).to_dict()}"
    return call_llm(prompt)

def generate_distribution_notice(df: pd.DataFrame) -> str:
    prompt = f"Create a distribution notice for LPs using this data: {df.head(3).to_dict()}"
    return call_llm(prompt)

def compute_fund_metrics(df: pd.DataFrame) -> str:
    prompt = f"Compute IRR, MOIC, DPI, and explain results from: {df.head(3).to_dict()}"
    return call_llm(prompt)

def write_lp_update(df: pd.DataFrame) -> str:
    prompt = f"Write a quarterly LP update letter using: {df.head(3).to_dict()}"
    return call_llm(prompt)

def simulate_gp_lp_qa(question: str, context: str) -> str:
    prompt = f"Fund context: {context}\nLP asks: {question}\nProvide a clear GP answer."
    return call_llm(prompt)

# UI

def main():
    st.set_page_config("FundPilot AI", page_icon="🚀", layout="wide")
    st.title("🚀 FundPilot AI – End-to-End Fund Management Copilot")
    st.write("Upload fund data, generate memos, calls, distributions, LP updates, and legal summaries.")

    uploaded_file = st.file_uploader("Upload fund-related CSV (deal or cash flow data)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("Data uploaded.")
    else:
        df = pd.DataFrame()

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📝 Deal Memo",
        "📁 Legal Summary",
        "💵 Capital Call",
        "💸 Distribution",
        "📈 Fund Metrics",
        "📨 LP Update",
        "💬 Q&A"
    ])

    with tab1:
        st.subheader("📝 Generate Deal Memo")
        if st.button("Create Memo"):
            if df.empty:
                st.error("Upload data first.")
            else:
                memo = generate_deal_memo(df)
                st.text_area("Deal Memo", value=memo, height=400)

    with tab2:
        st.subheader("📁 Legal Doc Summarizer")
        doc = st.text_area("Paste legal document text")
        if st.button("Summarize Document"):
            if not doc:
                st.error("Paste the document.")
            else:
                summary = summarize_legal_doc(doc)
                st.text_area("Summary", value=summary, height=300)

    with tab3:
        st.subheader("💵 Capital Call Notice")
        if st.button("Generate Capital Call"):
            if df.empty:
                st.error("Upload fund data.")
            else:
                out = generate_capital_call(df)
                st.text_area("Capital Call", value=out, height=300)

    with tab4:
        st.subheader("💸 LP Distribution Notice")
        if st.button("Generate Distribution"):
            if df.empty:
                st.error("Upload distribution data.")
            else:
                notice = generate_distribution_notice(df)
                st.text_area("Distribution Notice", value=notice, height=300)

    with tab5:
        st.subheader("📈 Fund Metrics Dashboard")
        if st.button("Run Metrics"):
            if df.empty:
                st.error("Upload data.")
            else:
                metrics = compute_fund_metrics(df)
                st.text_area("Metrics", value=metrics, height=300)

    with tab6:
        st.subheader("📨 LP Update Letter")
        if st.button("Generate Update"):
            if df.empty:
                st.error("Upload data.")
            else:
                update = write_lp_update(df)
                st.text_area("LP Update", value=update, height=400)

    with tab7:
        st.subheader("💬 GP/LP Question Simulator")
        context = st.text_area("Paste fund background or context")
        q = st.text_input("LP-style question")
        if st.button("Answer as GP"):
            if not context or not q:
                st.error("Provide context and question.")
            else:
                answer = simulate_gp_lp_qa(q, context)
                st.markdown(f"**GP Answer:** {answer}")

if __name__ == "__main__":
    main()
