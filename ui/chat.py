import streamlit as st
import requests
    

st.title("Dawa AI")

if "messages" not in st.session_state:
    st.session_state.messages = []
    

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if query := st.chat_input("How may I assist you today?"):
    
    with st.chat_message("user"):
        st.write(query)
    st.session_state.messages.append({"role": "user", "content": query})

    data = {"query": query}
    res = requests.post(f"http://localhost:8000/chats/",json=data)
    
    res_text = res.json()
    
    with st.chat_message("assistant"):
        
        st.markdown(res_text)

    st.session_state.messages.append({"role": "assistant", "content": res_text})
