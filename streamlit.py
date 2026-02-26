import streamlit as st
import requests

st.set_page_config(page_title="RAG Chatbot", layout="centered")

BACKEND = "http://localhost:8000"


def get_chatbot_response(query):
    try:
        response = requests.post(f"{BACKEND}/chat", json={"text": query}, timeout=60)
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        if hasattr(e, "response") and getattr(e.response, "text", None):
            return f"Error: {e.response.text}"
        return f"Error: {str(e)}"


st.sidebar.title("Knowledge Base")

if st.sidebar.button("Reset Knowledge Base"):
    try:
        res = requests.post(f"{BACKEND}/reset", json={})
        if res.status_code == 200:
            st.sidebar.success("Knowledge base reset.")
        else:
            st.sidebar.error(f"Reset failed: {res.text}")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

uploaded = st.sidebar.file_uploader(
    "Upload a PDF/CSV/XLSX/TXT",
    type=["pdf", "csv", "xlsx", "txt"]
)

if uploaded and st.sidebar.button("Upload & Index"):
    ext = uploaded.name.split(".")[-1].lower()
    mime = {
        "pdf": "application/pdf",
        "csv": "text/csv",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "txt": "text/plain",
    }.get(ext, "application/octet-stream")

    files = {"file": (uploaded.name, uploaded.getvalue(), mime)}
    try:
        res = requests.post(f"{BACKEND}/upload", files=files, timeout=120)
        if res.status_code == 200:
            st.sidebar.success("Uploaded & indexed.")
        else:
            st.sidebar.error(f"Upload failed: {res.text}")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

st.title("RAG Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What would you like to know?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    response = get_chatbot_response(prompt)
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
