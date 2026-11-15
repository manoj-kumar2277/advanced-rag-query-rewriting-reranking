"""Streamlit UI:  streamlit run app.py"""
import streamlit as st

import rag

st.set_page_config(page_title="Advanced RAG", layout="wide")
st.title("Advanced RAG: Query Rewriting + Reranking")

mode = st.sidebar.radio("Retrieval mode", rag.MODES, index=3)
compare = st.sidebar.checkbox("Compare basic vs enhanced side by side")
question = st.text_input("Ask a question about your documents")


def show(title, m):
    st.subheader(title)
    res = rag.retrieve(question, m)
    st.caption(f"Search query: **{res['search_query']}**  |  "
               f"retrieval {res['t_retrieval']:.2f}s")
    text, t = rag.answer(question, res["chunks"])
    st.markdown(text)
    st.caption(f"Answer generated in {t:.1f}s")
    with st.expander("Retrieved chunks"):
        for i, c in enumerate(res["chunks"], 1):
            st.markdown(f"**[{i}] {c['source']} p.{c['page']}**  (score {c['score']:.3f})")
            st.write(c["text"])


if question:
    if compare:
        a, b = st.columns(2)
        with a:
            show("Basic RAG", "basic")
        with b:
            show("Enhanced RAG (rewrite + rerank)", "both")
    else:
        show(f"Mode: {mode}", mode)
