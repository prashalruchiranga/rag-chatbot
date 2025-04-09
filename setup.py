import os
from dotenv import load_dotenv
import json
import vertexai
from langchain.chat_models import init_chat_model
from langchain_google_vertexai import VertexAIEmbeddings
import asyncio
from pathlib import Path
from chatbot_engine import ChatSession
from vector_db import VectorDBCreator


class SetupChatbot:
    def __init__(self, uploaded_files):
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
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        os.environ["GOOGLE_CLOUD_PROJECT"] = os.getenv("GOOGLE_CLOUD_PROJECT")
        os.environ["GOOGLE_CLOUD_LOCATION"] = os.getenv("GOOGLE_CLOUD_LOCATION")
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = os.getenv("GOOGLE_GENAI_USE_VERTEXAI")

    def initialize_model(self):
        vertexai.init(
            project=os.environ["GOOGLE_CLOUD_PROJECT"],
            location=os.environ["GOOGLE_CLOUD_LOCATION"]
        )
        self.llm = init_chat_model(
            self.config["model"]["name"],
            model_provider=self.config["model"]["provider"]
        )

    async def create_vector_store(self):
        vector_db_creator = VectorDBCreator(
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

    # # Example usage of the session (streaming a message)
    # thread = "abc12345"
    # while True:
    #     query = input("query: ")
    #     # for msg in session.stream_values(thread_id=thread, message=query):
    #     #     if msg.type == "ai":
    #     #         print(f"{msg.content.strip()}")
    #     message_response = session.send_message(thread_id=thread, message=query)
    #     print(f"{message_response.content}\n")
