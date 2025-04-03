from langchain_community.document_loaders import PyPDFLoader
from typing import List
from langchain_core.documents import Document
import re


async def Load_PDF(file_path: str) -> List:
    loader = PyPDFLoader(file_path, extraction_mode="layout")
    pdf_pages = []
    async for page in loader.alazy_load():
        pdf_pages.append(page)
    return pdf_pages


def filter_text(file_path, template, model, pages):
    # This clears the file content
    with open(file_path, "w") as file:
        pass  
    for page in pages:
        prompt = template.invoke({"page": page.page_content})
        response = model.invoke(prompt)
        with open(file_path, "a") as file:
            content = replace_red_char(response.content)
            # Add a space after content
            file.write(content + " ")
    return file_path


def replace_red_char(text):
    # Replace any sequence of spaces, and\or new line characters with a single space
    result = re.sub(r'[\s\n]+', ' ', text)
    return result.strip(" ")


def create_db(vertextai_embedding_model, dimension, splits, vector_store):
    if vertextai_embedding_model == "text-embedding-005":
        assert dimension == 768
        record_ids = vector_store.add_documents(documents=splits)
    return record_ids


def update_metadata(field: str, value: str, splits: List[Document]) -> List[Document]:
    for split in splits:
        split.metadata[field] = value
    return splits


