import streamlit as st
import random
import time
from setup import setup

# App title
st.set_page_config(page_title="Gemini RAG-Chatbot")

# Display a caption
st.caption("Powered by Langchain, VertexAI, FAISS and Streamlit")

# Sidebar UI
st.sidebar.header("🤖 RAG-Chatbot")
st.sidebar.write("Chat with your documents! Combined with RAG to deliver accurate and context-aware responses.")

st.sidebar.subheader("API Key")
api_key = st.sidebar.text_input("Enter your HuggingFace API key", type="password")

st.sidebar.subheader("Models and Parameters")
selected_model = st.sidebar.selectbox(
    "Select preferred model",
    ("gemini-2.5-pro-exp-03-25", "gemini-2.0-flash", "gemini-2.5-pro-preview-03-25",
     "gemini-2.0-flash-lite", "gemini-2.0-flash-thinking-exp-01-21")
)
documents = st.sidebar.file_uploader(label="Upload your files", accept_multiple_files=True)

# Initialize session state variables
if "chat_session" not in st.session_state:
    st.session_state.chat_session = None
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "success_message" not in st.session_state:
    st.session_state.success_message = False

# Function to clear chat history
def clear_chat():
    st.session_state.messages = []

success_placeholder = st.empty()

# Create DB on button click
if st.sidebar.button(label="Submit Files", type="primary"):
    # Clear the chat history
    clear_chat()
    
    with st.spinner(text="Creating the Vector Database...", show_time=True):
        chat_session = setup()
        st.session_state.chat_session = chat_session
        success_placeholder.success("✅ Database successfully created. You may now chat.")

# Show success message if needed (and handle auto-clear)
# if st.session_state.success_message:
#     success_placeholder = st.empty()
#     success_placeholder.success("✅ Database successfully created. You may now chat.")

# Container for chat messages
chat_container = st.container()

# Show chat history in the container
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input logic based on DB status
if st.session_state.chat_session is not None:
    prompt = st.chat_input("What is up?", disabled=False)
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                with st.spinner("Thinking"):
                    assistant_response = st.session_state.chat_session.send_message(thread_id="12345", message=prompt).content
                for chunk in assistant_response.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
else:
    st.chat_input("Please submit files to create the vector database first to start chatting.", disabled=True)


