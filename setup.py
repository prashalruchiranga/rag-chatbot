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


def setup():
    # Define file paths
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.joinpath("database", "data")
    config_path = script_dir.joinpath("config.json")
    log_path = script_dir.joinpath("logs.log")
    
    # Load config
    with open(config_path, "r") as file:
        config = json.load(file)

    # Load .env
    dotenv_path = os.path.join(script_dir, "../.env")

    # Load environment variables
    load_dotenv(dotenv_path)
    GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
    GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")
    GOOGLE_GENAI_USE_VERTEXAI = os.getenv("GOOGLE_GENAI_USE_VERTEXAI")

    # Configure the language model
    vertexai.init(project=GOOGLE_CLOUD_PROJECT, location=GOOGLE_CLOUD_LOCATION)
    llm = init_chat_model(config["model"]["name"], model_provider=config["model"]["provider"])

    # Create vector database
    vector_db_creator = VectorDBCreator(config_file_path=config_path, data_directory=data_dir, log_file_path=log_path)
    vector_store = asyncio.run(vector_db_creator.create())

    # Initialize a ChatSession
    session = ChatSession(model=llm, vector_store=vector_store)
    return session

    # # Example usage of the session (streaming a message)
    # thread = "abc12345"
    # while True:
    #     query = input("query: ")
    #     # for msg in session.stream_values(thread_id=thread, message=query):
    #     #     if msg.type == "ai":
    #     #         print(f"{msg.content.strip()}")
    #     message_response = session.send_message(thread_id=thread, message=query)
    #     print(f"{message_response.content}\n")



