import streamlit as st
import time
from setup import SetupChatbot

# Clear previous chats
def clear_chat():
    st.session_state.messages = []
    welcome_placeholder.empty()

# Initialize session state variables
if "chatbot" not in st.session_state:
    st.session_state.chatbot = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "files_submitted" not in st.session_state:
    st.session_state.files_submitted = False

# App title
st.set_page_config(page_title="Gemini RAG-Chatbot")
# Display a caption
st.caption("Powered by Langchain, VertexAI, FAISS and Streamlit")
# Welcome text
welcome_placeholder = st.header("What can I help with?")

# Placeholders
success_placeholder = st.empty()
info_placeholder = st.empty()

# Container for chat messages
chat_container = st.container()

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
st.session_state.files_submitted = False
uploaded_files = st.sidebar.file_uploader(label="Upload your files", accept_multiple_files=True, type=["pdf"])
if uploaded_files:
    st.session_state.files_submitted = True

# Create DB on button click
if st.sidebar.button(label="SUBMIT", type="primary", disabled=not st.session_state.files_submitted):
    clear_chat()
    with st.spinner(text="Working on your files...", show_time=True):
        setup = SetupChatbot(uploaded_files=uploaded_files)
        chatbot = setup.setup()
    st.session_state.chatbot = chatbot
    success_placeholder.success("✅ You may now chat with your documents")

# Show chat history in the container
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# User, assistant model conversations
if st.session_state.chatbot is not None:
    prompt = st.chat_input("Whats up?", disabled=False)
    if prompt:
        welcome_placeholder.empty()
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                with st.spinner("Thinking"):
                    assistant_response = st.session_state.chatbot.send_message(message=prompt).content
                for chunk in assistant_response.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
else:
    st.chat_input("Whats up?", disabled=True)
    info_placeholder.info("ℹ️ Please submit documents to start chatting with them")


