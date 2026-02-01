import streamlit as st
from lib.gemini import gemini_ai
from runners import run_hybrid_search


def hybrid_stream(query: str, limit: int = 5):
    """Generator for Streamlit"""
    results = run_hybrid_search(query, limit)

    for idx, result in enumerate(results, 1):
        res = result[1]
        yield f"{idx}. Medicine: {res['name']}\n"
        yield f"   Section: {res['section']}\n"
        yield f"   Content: {res['text'][:100]}\n\n"


def ai_question(query: str):
    results = run_hybrid_search(query)
    yield gemini.question(query, results)


st.title("Dawa AI")
gemini = gemini_ai("gemini-2.5-flash")

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What's up ?"):
    with st.chat_message("user"):
        st.write(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        response = st.write_stream(ai_question(query=prompt))

    st.session_state.messages.append({"role": "assistant", "content": response})
