import os
from dotenv import load_dotenv
import vertexai
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
import json
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import asyncio
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from pinecone import ServerlessSpec
import time
import logging
from utilities import Load_PDF, create_db, filter_text, update_metadata


### Define file paths
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "../", "config.json")
with open(config_path, "r") as file:
    config = json.load(file)

dotenv_path = os.path.join(script_dir, "../.env")
pdf_path = os.path.join(script_dir, "data", config["pdf_path"])
output_text_path = os.path.join(script_dir, "data", config["output_text_path"])
saved_prompts = os.path.join(script_dir, "../", config["saved_prompts"])
log_path = os.path.join(script_dir, "../", config["logs"])


### Load environment variables
load_dotenv(dotenv_path)
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")
GOOGLE_GENAI_USE_VERTEXAI = os.getenv("GOOGLE_GENAI_USE_VERTEXAI")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


# Configure logging
logging.basicConfig(
    filename=log_path,   
    level=logging.INFO,    
    format="%(asctime)s - %(levelname)s - %(message)s"
)


### Configure the Language Model
vertexai.init(project=GOOGLE_CLOUD_PROJECT, location=GOOGLE_CLOUD_LOCATION) 
gemini = init_chat_model("gemini-2.0-flash-001", model_provider="google_vertexai")


### Extract text from the pdf, filter and save them to a text file
pages = asyncio.run(Load_PDF(file_path=pdf_path))
with open(saved_prompts, "r") as file:
    prompts = json.load(file)


system_template = "\n\n".join(prompts["system_template_text_preprocessing"])
prompt_template = ChatPromptTemplate.from_messages([("system", system_template), ("user", "{page}")])  
_ = filter_text(output_text_path, prompt_template, gemini, pages)


### Load filtered text file and split it to Document objects
loader = TextLoader(output_text_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  
    chunk_overlap=400,  
    add_start_index=True,  
    separators=[r'CHAPTER\s+[IVXLCDM]*\s',r'\d+\.\s'],
    is_separator_regex=True,
    keep_separator=True
)
all_splits = text_splitter.split_documents(docs)
logging.info(f"Split the text file post into {len(all_splits)} sub-documents.")


### Update source field in metadata 
all_splits = update_metadata("source", config["pdf_path"], all_splits)


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


