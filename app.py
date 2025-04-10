import streamlit as st
import time
from setup import SetupChatbot

# Clear previous chats
def clear_chat():
    st.session_state.messages = []

# Initialize session state variables
if "session" not in st.session_state:
    st.session_state.session = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "files_submitted" not in st.session_state:
    st.session_state.files_submitted = False
if "working_model" not in st.session_state:
    st.session_state.working_model = None
if "setup" not in st.session_state:
    st.session_state.setup = None

# App title
st.set_page_config(page_title="Gemini RAG-Chatbot")
# Display a caption
st.caption("Powered by Langchain, Google AI Studio, FAISS and Streamlit")

# Welcome text
# welcome_placeholder = st.header("What can I help with?")

# Placeholders
info_placeholder = st.empty()
created_new_session = st.empty()
success_placeholder = st.empty()

# Container for chat messages
chat_container = st.container()

# Sidebar UI
st.sidebar.header("🤖 RAG-Chatbot")
st.sidebar.write("Chat with your documents! Combined with RAG to deliver accurate and context-aware responses.")
st.sidebar.subheader("API Key")
api_key = st.sidebar.text_input("Enter your Google AI Studio API key", type="password")
st.sidebar.subheader("Models and Parameters")
selected_model = st.sidebar.selectbox(
    "Select preferred model",
    (
        "gemini-2.5-pro-preview-03-25", 
        "gemini-2.0-flash", 
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash", 
        # "gemini-1.5-flash-8b", 
        "gemini-1.5-pro")
    )
st.session_state.files_submitted = False
uploaded_files = st.sidebar.file_uploader(label="Upload your files", accept_multiple_files=True, type=["pdf"])
if uploaded_files:
    st.session_state.files_submitted = True

# Create DB on button click
if st.sidebar.button(label="SUBMIT", disabled=not st.session_state.files_submitted):
    clear_chat()
    with st.spinner(text="Working on your files...", show_time=True):
        setup = SetupChatbot(model=selected_model, api_key=api_key, uploaded_files=uploaded_files)
        session = setup.setup()
    st.session_state.setup = setup
    st.session_state.session = session
    st.session_state.working_model = selected_model
    created_new_session.info("ℹ️ Created a new chat session")
    success_placeholder.success("✅ You may now chat with your documents")

if (selected_model != st.session_state.working_model) and (st.session_state.session != None):
    new_session = st.session_state.setup.update_setup(new_model=selected_model)
    st.session_state.working_model = selected_model
    st.session_state.session = new_session
    st.session_state.messages = []
    created_new_session.info("ℹ️ Created a new chat session")

# Show chat history in the container
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# User, assistant model conversations
if st.session_state.session is not None:
    prompt = st.chat_input("Whats up?", disabled=False)
    if prompt:
        created_new_session.empty()
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                with st.spinner(f"**{st.session_state.working_model}** is thinking"):
                    assistant_response = st.session_state.session.send_message(message=prompt).content
                for chunk in assistant_response.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
else:
    st.chat_input("Whats up?", disabled=True)
    info_placeholder.info("ℹ️ Please submit at least one document to start chatting")
