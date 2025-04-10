import os
from dotenv import load_dotenv
import json
from langchain_google_genai import ChatGoogleGenerativeAI
import asyncio
from pathlib import Path
from chatbot_engine import ChatSession
from vector_db import VectorDBCreator


class SetupChatbot:
    def __init__(self, model, api_key, uploaded_files):
        self.model = model
        self.api_key = api_key
        self.uploaded_files = uploaded_files
        self.script_dir = Path(__file__).resolve().parent
        self.config_path = self.script_dir.joinpath("config.json")
        self.log_path = self.script_dir.joinpath("logs.log")
        self.config = None
        self.llm = None
        self.vector_store = None
        self.session = None

    def load_config(self):
        with open(self.config_path, "r") as file:
            self.config = json.load(file)

    def load_environment_variables(self):
        dotenv_path = os.path.join(self.script_dir, "../.env")
        load_dotenv(dotenv_path)

    def initialize_model(self):
        self.llm = ChatGoogleGenerativeAI(
            model=self.model,
            temperature=0,
            google_api_key=self.api_key
            )

    async def create_vector_store(self):
        vector_db_creator = VectorDBCreator(
            api_key=self.api_key,
            source_data_files=self.uploaded_files,
            config_file_path=self.config_path,
            log_file_path=self.log_path
        )
        self.vector_store = await vector_db_creator.create()

    def initialize_session(self):
        self.session = ChatSession(model=self.llm, vector_store=self.vector_store)

    def setup(self):
        self.load_config()
        self.load_environment_variables()
        self.initialize_model()
        asyncio.run(self.create_vector_store())
        self.initialize_session()
        return self.session

    def update_setup(self, new_model):
        '''Creates a new chat session with updated model and previous vector store'''
        self.model = new_model
        self.initialize_model()
        self.initialize_session()
        return self.session   

    # # Example usage of the session (streaming a message)
    # thread = "abc12345"
    # while True:
    #     query = input("query: ")
    #     # for msg in session.stream_values(thread_id=thread, message=query):
    #     #     if msg.type == "ai":
    #     #         print(f"{msg.content.strip()}")
    #     message_response = session.send_message(thread_id=thread, message=query)
    #     print(f"{message_response.content}\n")
