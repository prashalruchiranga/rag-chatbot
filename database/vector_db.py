import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
import json
from pdf_processor import PDFProcessor


class VectorDBCreator:
    def __init__(self, config_file_path, data_directory, log_file_path):
        self.config_path = config_file_path
        self.data_dir = data_directory
        self.log_path = log_file_path
        self._load_config()
        self._setup_logging()

    def _load_config(self):
        with open(self.config_path, "r") as file:
            self.config = json.load(file)

    def _setup_logging(self):
        logging.basicConfig(
            filename=self.log_path,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

    async def create(self):
        # Process pdf files 
        pdf_processor = PDFProcessor(data_directory=self.data_dir)
        await pdf_processor.process_pdfs_in_directory()
        docs = pdf_processor.load_txts_in_directory()

        # Split text files to Documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
        all_splits = text_splitter.split_documents(docs)
        logging.info(f"Split text into {len(all_splits)} sub-documents.")
        # print(f"Split text into {len(all_splits)} sub-documents.")

        # Define embeddings model
        embeddings = VertexAIEmbeddings(model=self.config["embeddings"]["model"])

        # Create vector database
        index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
        vector_store = FAISS(
            embedding_function=embeddings, 
            index=index,
            docstore=InMemoryDocstore(), 
            index_to_docstore_id={}
        )

        # Index chunks
        record_ids = vector_store.add_documents(documents=all_splits)

        # Log created ids
        for id in record_ids:
            logging.info(f"Created id: {id}")
            # print(f"Created id: {id}")

        return vector_store
    

