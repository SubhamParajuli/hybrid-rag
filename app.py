import streamlit as st
import requests

st.set_page_config(
    page_title="Hybrid RAG Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Hybrid RAG Assistant")
st.markdown("Powered by **LangChain**, **Chroma**, **BM25**, **Ollama**, and **FastAPI**.")

# Explicit local loopback address to avoid WinError network resolution quirks
API_URL = "http://127.0.0.1:8000"

# --- Sidebar: Document Management & Uploads ---
with st.sidebar:
    st.header("📂 Document Management")
    st.markdown("Upload files directly to the API server.")

    uploaded_files = st.file_uploader(
        "Upload new documents",
        type=["txt", "pdf", "md", "docx"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Upload & Ingest", type="primary"):
            with st.spinner("Uploading files and indexing vector store..."):
                try:
                    # Construct multipart payload for multiple files
                    files_payload = [
                        ("files", (file.name, file.getvalue(), file.type)) 
                        for file in uploaded_files
                    ]
                    response = requests.post(
                        f"{API_URL}/upload-and-ingest", 
                        files=files_payload
                    )
                    
                    if response.status_code == 200:
                        st.success("Files successfully uploaded and indexed!")
                    else:
                        err = response.json().get("detail", "Unknown error")
                        st.error(f"Ingestion failed: {err}")
                except Exception as e:
                    st.error(f"Failed to connect to FastAPI: {e}")

    st.divider()
    if st.button("Clear Conversation History"):
        st.session_state.messages = []
        st.rerun()

# --- Main Chat UI with Persistence ---
st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input Bar
if prompt := st.chat_input("Ask a question about your documents..."):
    # 1. Display user query immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Query FastAPI passing current question + full conversation history
    with st.chat_message("assistant"):
        with st.spinner("Thinking (Searching hybrid retriever + Ollama)..."):
            try:
                payload = {
                    "question": prompt,
                    "chat_history": st.session_state.messages[:-1] # Send prior history excluding current prompt
                }
                
                response = requests.post(f"{API_URL}/ask", json=payload)
                
                if response.status_code == 200:
                    answer = response.json().get("answer", "No answer returned.")
                    st.markdown(answer)
                    
                    # Update session state with assistant response
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    err = response.json().get("detail", "Unknown error")
                    st.error(f"Server Error: {err}")
                    
            except Exception as e:
                st.error(f"Could not connect to FastAPI server at `{API_URL}`. Make sure Uvicorn is running!")