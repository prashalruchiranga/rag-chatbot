import os
from dotenv import load_dotenv
import json
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from pinecone import ServerlessSpec
import time
import logging
from pathlib import Path
from utilities import create_db, update_metadata


### Define file paths
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "../", "config.json")
with open(config_path, "r") as file:
    config = json.load(file)

dotenv_path = os.path.join(script_dir, "../.env")
data_dir = os.path.join(script_dir, "data")
log_dir = os.path.join(script_dir, "../", "logs")
database_log = os.path.join(log_dir, config["logging"]["database"])


### Load environment variables
load_dotenv(dotenv_path)
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")
GOOGLE_GENAI_USE_VERTEXAI = os.getenv("GOOGLE_GENAI_USE_VERTEXAI")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


### Configure logging
logging.basicConfig(
    filename=database_log,   
    level=logging.INFO,    
    format="%(asctime)s - %(levelname)s - %(message)s"
)


### Open txt files
directory = Path(data_dir)
txt_files = [f.name for f in directory.glob("*.txt")]


### Load filtered text file and split it to Document objects
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  
    chunk_overlap=400,  
    add_start_index=True,  
    separators=[r'CHAPTER\s+[IVXLCDM]*\s',r'\d+\.\s'],
    is_separator_regex=True,
    keep_separator=True
)

for file in txt_files:
    txt_path = os.path.join(data_dir, file)
    loader = TextLoader(txt_path)
    docs = loader.load()
    all_splits = text_splitter.split_documents(docs)
    logging.info(f"Split the text file {txt_path} into {len(all_splits)} sub-documents.")
    # Update source field in metadata 
    all_splits = update_metadata("source", file, all_splits)


### Add Documents to Pinecone vector db
embedding_model = config["embeddings"]["model"]
vector_dimension = config["embeddings"]["dimension"]
index_name = config["index"]["name"]
namespace = config["index"]["namespace"]
cloud_provider = config["index"]["cloud_provider"]
aws_region = config["index"]["region"]
similarity_metric = config["index"]["similarity_metric"]

embeddings = VertexAIEmbeddings(model=embedding_model)
pinecone = Pinecone(api_key=PINECONE_API_KEY)

if index_name not in [index_info["name"] for index_info in pinecone.list_indexes()]:
    pinecone.create_index(
        name=index_name,
        dimension=vector_dimension,
        metric=similarity_metric,
        spec=ServerlessSpec(cloud=cloud_provider, region=aws_region)
        )
    logging.info(f"Created index {index_name}")
    while not pinecone.describe_index(index_name).status["ready"]:
        time.sleep(1)
else:
    raise Exception(f"Index {index_name} already exists")

vector_store = PineconeVectorStore(embedding=embeddings, index=pinecone.Index(index_name), namespace=namespace)
ids = create_db(vertextai_embedding_model=embedding_model, dimension=vector_dimension, splits=all_splits, vector_store=vector_store)


### Log information
for id in ids:
    logging.info(f"Created record: {id}") 
log = f"Inserted {len(ids)} records."
logging.info(log)
print(f"{log} See the log file for record ids.")


