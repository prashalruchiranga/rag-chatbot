import streamlit as st
from setup import SetupChatbot

# Clear previous chats
def clear_chat():
    st.session_state.messages = []

# Initialize session state variables
if "session" not in st.session_state:
    st.session_state.session = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "files_uploaded" not in st.session_state:
    st.session_state.files_uploaded = False
if "working_model" not in st.session_state:
    st.session_state.working_model = None
if "setup" not in st.session_state:
    st.session_state.setup = None
if "api_key" not in st.session_state:
    st.session_state.api_key = None

# App title
st.set_page_config(page_title="RAG-Chatbot")
st.markdown("<h1 style='font-size: 30px;'>🤖 RAG-CHATBOT</h1>", unsafe_allow_html=True)

# Display a caption
st.caption("Powered by Langchain, Google AI Studio, FAISS and Streamlit")

# Placeholders
info_placeholder = st.empty()
created_new_session = st.empty()
success_placeholder = st.empty()

# Container for chat messages
chat_container = st.container()

# Sidebar UI
st.sidebar.header("Main Menu")
st.sidebar.write("Chat with your documents! Combined with RAG to deliver accurate and context-aware responses.")
st.sidebar.subheader("Models and Documents")
selected_model = st.sidebar.selectbox(
    "Select preferred model",
    (
        "gemini-2.5-pro-preview-03-25", 
        "gemini-2.0-flash", 
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash", 
        "gemini-1.5-pro")
    )
st.session_state.files_uploaded = False
uploaded_files = st.sidebar.file_uploader(label="Upload your files", accept_multiple_files=True, type=["pdf"])
if uploaded_files:
    st.session_state.files_uploaded = True

if st.session_state.files_uploaded == False:
    st.info("ℹ️ Please submit at least one document to start chatting")

# Create DB on button click
if st.sidebar.button(label="SUBMIT", disabled=not st.session_state.files_uploaded):
    try:
        clear_chat()
        with st.spinner(text="Working on your files...", show_time=True):
            setup = SetupChatbot(model=selected_model, api_key=st.session_state.api_key, uploaded_files=uploaded_files)
            session = setup.setup()   
        st.session_state.setup = setup
        st.session_state.session = session
        st.session_state.working_model = selected_model
        created_new_session.info(f"ℹ️ Created a new chat session with {st.session_state.working_model}")
        success_placeholder.success("✅ You may now chat with your documents")
    except:
        st.error("❌ Invalid API Key")

# Gemini API Key
st.sidebar.subheader("API Key")
api_key = st.sidebar.text_input("Enter your Google AI Studio API key", type="password", disabled=bool(st.session_state.session))
if bool(api_key) != False:
    st.session_state.api_key = api_key

if (selected_model != st.session_state.working_model) and (st.session_state.session != None):
    new_session = st.session_state.setup.update_setup(new_model=selected_model)
    st.session_state.working_model = selected_model
    st.session_state.session = new_session
    st.session_state.messages = []
    created_new_session.info(f"ℹ️ Created a new chat session with {st.session_state.working_model}")
    success_placeholder.success("✅ You may now chat with your documents")

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
                with st.spinner("Thinking"):
                    for chunk in st.session_state.session.stream_values(message=prompt):
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
else:
    st.chat_input("Whats up?", disabled=not bool(st.session_state.session))
