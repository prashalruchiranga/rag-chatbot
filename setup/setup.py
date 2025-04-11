import os
from dotenv import load_dotenv
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai._common import GoogleGenerativeAIError
from google.auth.exceptions import DefaultCredentialsError
import asyncio
from pathlib import Path
from chatbot.chatbot_engine import ChatSession
from database.vector_db import VectorDBCreator


class SetupChatbot:
    def __init__(self, model, api_key, uploaded_files):
        self.model = model
        self.api_key = api_key
        self.uploaded_files = uploaded_files
        self.script_dir = Path(__file__).resolve().parent
        self.config_path = self.script_dir.joinpath("../", "config.json")
        self.log_path = self.script_dir.joinpath("../", "logs.log")
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
            temperature=self.config["temperature"],
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
        try:
            self.initialize_model()
            asyncio.run(self.create_vector_store())
        except (DefaultCredentialsError, GoogleGenerativeAIError) as e:
            raise ValueError(f"Invalid API key")
        self.initialize_session()
        return self.session

    def update_setup(self, new_model):
        self.model = new_model
        try:
            self.initialize_model()
        except (DefaultCredentialsError, GoogleGenerativeAIError) as e:
            raise ValueError(f"Invalid API key")
        self.initialize_session()
        return self.session
    