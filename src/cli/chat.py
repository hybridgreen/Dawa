import streamlit as st
from lib.gemini import gemini_ai
from runners import run_hybrid_search
from google.genai import types

def hybrid_stream(query: str, limit: int = 5):
    """Generator for Streamlit"""
    results = run_hybrid_search(query, limit)

    for idx, result in enumerate(results, 1):
        res = result[1]
        yield f"{idx}. Medicine: {res['name']}\n"
        yield f"   Section: {res['section']}\n"
        yield f"   Content: {res['text'][:100]}\n\n"
    

st.title("Dawa AI")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "gemini_chat" not in st.session_state:
    
    gemini = gemini_ai("gemini-2.5-flash")

    st.session_state.gemini_chat = gemini.client.chats.create(
        model="gemini-2.5-flash",
        config= types.GenerateContentConfig(
                system_instruction= gemini.system_prompt
            ))
    

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if query := st.chat_input("How may I assist you today?"):
    with st.chat_message("user"):
        st.write(query)

    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        documents = run_hybrid_search(query)
        
        prompt = f"""Answer the following question based on the provided documents.

    Question: {query}

    Documents:
    {documents}

    General instructions:
    
    Guidance on types of questions:
    - Factual questions: Provide a direct answer
    - Analytical questions: Compare and contrast information from the documents
    - Opinion-based questions: Acknowledge subjectivity and provide a balanced view

    Answer:"""
        
        response = st.session_state.gemini_chat.send_message(prompt)
        
        st.markdown(response.text)

    st.session_state.messages.append({"role": "assistant", "content": response.text})
