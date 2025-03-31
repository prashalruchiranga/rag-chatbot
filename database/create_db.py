import os
from dotenv import load_dotenv
import vertexai
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
import json
from langchain.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import asyncio
from utilities import Load_PDF, create_db, filter_text


### Load environment variables
load_dotenv()
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")
GOOGLE_GENAI_USE_VERTEXAI = os.getenv("GOOGLE_GENAI_USE_VERTEXAI")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


### Define file paths
pdf_path = ("data/constitution-upto-18th.pdf")
output_text_path = ("data/constitution-upto-18th-filtered.text")


### Configure the Language Model
vertexai.init(project=GOOGLE_CLOUD_PROJECT, location=GOOGLE_CLOUD_LOCATION) 
gemini = init_chat_model("gemini-2.0-flash-001", model_provider="google_vertexai")


### Extract text from the pdf, filter and save them to a text file
pages = asyncio.run(Load_PDF(file_path=pdf_path))
with open("prompts.json", "r") as file:
    prompts = json.load(file)

system_template = "\n\n".join(prompts["system_template_text_preprocessing"])
prompt_template = ChatPromptTemplate.from_messages([("system", system_template), ("user", "{page}")])  

l_limit = 18
u_limit = 25
# high = 173
relevant_pages = pages[l_limit:u_limit] # Remove unncessary information. 
_ = filter_text(output_text_path, prompt_template, gemini, relevant_pages)


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
print(f"Split the text file post into {len(all_splits)} sub-documents.")


### Add Documents to Pinecone vector db
embedding_model = "text-embedding-005"
_ = create_db(
    vertextai_embedding_model_name=embedding_model, 
    dimension=768, 
    pinecone_index="data-vectors", 
    namespace="Acts", 
    pinecone_api_key=PINECONE_API_KEY, 
    splits=all_splits
)
