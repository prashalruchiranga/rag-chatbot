import os
from dotenv import load_dotenv
import vertexai
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
import json
import asyncio
import logging
from pathlib import Path
from utilities import Load_PDF, filter_text


### Define file paths
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "../", "config.json")
with open(config_path, "r") as file:
    config = json.load(file)

dotenv_path = os.path.join(script_dir, "../", ".env")
data_dir = os.path.join(script_dir, "data")
saved_prompts = os.path.join(script_dir, "../", config["saved_prompts"])
log_dir = os.path.join(script_dir, "../", "logs")
preprocessor_log = os.path.join(log_dir, config["logging"]["preprocessor"])


### Load environment variables
load_dotenv(dotenv_path)
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")
GOOGLE_GENAI_USE_VERTEXAI = os.getenv("GOOGLE_GENAI_USE_VERTEXAI")


### Configure logging
logging.basicConfig(
    filename=preprocessor_log,   
    level=logging.INFO,    
    format="%(asctime)s - %(levelname)s - %(message)s"
)


### Configure the Language Model
vertexai.init(project=GOOGLE_CLOUD_PROJECT, location=GOOGLE_CLOUD_LOCATION) 
gemini = init_chat_model("gemini-2.0-flash-001", model_provider="google_vertexai")


### Load saved prompts
with open(saved_prompts, "r") as file:
    prompts = json.load(file)
system_template = "\n\n".join(prompts["system_template_text_preprocessing"])
prompt_template = ChatPromptTemplate.from_messages([("system", system_template), ("user", "{page}")]) 


### Open PDF files
directory = Path(data_dir)
pdf_files = [f.name for f in directory.glob("*.pdf")]


### Extract text from the PDFs, filter and save them to text files
for pdf in pdf_files:
    pdf_path = Path(os.path.join(data_dir, pdf))
    output_path = pdf_path.with_stem(pdf_path.stem + "-filtered").with_suffix(".txt")
    pages = asyncio.run(Load_PDF(file_path=pdf_path))
    log = filter_text(output_path, prompt_template, gemini, pages)
    logging.info(f"filtered text file created: {log}")


