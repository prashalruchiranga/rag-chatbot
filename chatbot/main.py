import os
from dotenv import load_dotenv
import json
import vertexai
from langchain.chat_models import init_chat_model
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from chatbot_engine import ChatSession

def main():
    # Define file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "../", "config.json")
    
    # Load config
    with open(config_path, "r") as file:
        config = json.load(file)

    dotenv_path = os.path.join(script_dir, "../.env")

    # Load environment variables
    load_dotenv(dotenv_path)
    GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
    GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")
    GOOGLE_GENAI_USE_VERTEXAI = os.getenv("GOOGLE_GENAI_USE_VERTEXAI")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

    # Configure the language model
    vertexai.init(project=GOOGLE_CLOUD_PROJECT, location=GOOGLE_CLOUD_LOCATION)
    llm = init_chat_model(config["model"]["name"], model_provider=config["model"]["provider"])

    # Define embedding model
    embeddings = VertexAIEmbeddings(model=config["embeddings"]["model"])

    # Define vector store
    index_name = config["index"]["name"]
    namespace = config["index"]["namespace"]
    pinecone = Pinecone(api_key=PINECONE_API_KEY)
    index = pinecone.Index(index_name)
    vector_store = PineconeVectorStore(embedding=embeddings, index=index, namespace=namespace)

    # Initialize a ChatSession
    session = ChatSession(model=llm, vector_store=vector_store)

    # Example usage of the session (streaming a message)
    session.stream_values(thread_id="abc12345", message="When will the president seat considered vacant?")
    
if __name__ == "__main__":
    main()


