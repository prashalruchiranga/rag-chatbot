import os
from dotenv import load_dotenv
import json
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
import asyncio
import logging
from pdf_processor import PDFProcessor


async def create_db():
    ### Define file paths
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.joinpath("data")
    config_path = script_dir.joinpath("../", "config.json")
    dotenv_path = script_dir.joinpath("../", ".env")
    log_path = script_dir.joinpath("../", "logs.log")

    ### Open configuration file
    with open(config_path, "r") as file:
        config = json.load(file)

    ### Load environment variables
    load_dotenv(dotenv_path)
    GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
    GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")
    GOOGLE_GENAI_USE_VERTEXAI = os.getenv("GOOGLE_GENAI_USE_VERTEXAI")

    ### Configure logging
    logging.basicConfig(
        filename=log_path,   
        level=logging.INFO,    
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    ### Process pdf files 
    pdf_processor = PDFProcessor(data_directory=data_dir)
    await pdf_processor.process_pdfs_in_directory()
    docs = pdf_processor.load_txts_in_directory()

    ### Split text files to Documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # chunk size (characters)
        chunk_overlap=200,  # chunk overlap (characters)
        add_start_index=True,  # track index in original document
    )
    all_splits = text_splitter.split_documents(docs)
    msg= f"Split text into {len(all_splits)} sub-documents."
    logging.log(msg=msg, level=logging.INFO)
    print(msg)

    ### Define embeddings model
    embeddings = VertexAIEmbeddings(model=config["embeddings"]["model"])

    ### Create vector database
    embedding_dim = len(embeddings.embed_query("hello world"))
    index = faiss.IndexFlatL2(embedding_dim)
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    ### Index chunks
    record_ids = vector_store.add_documents(documents=all_splits)

    ### Log created ids
    for id in record_ids:
        msg = f"Created id: {id}"
        logging.log(msg=msg, level=logging.INFO)
        print(msg)

asyncio.run(create_db())


