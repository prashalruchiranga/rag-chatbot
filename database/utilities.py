from langchain_community.document_loaders import PyPDFLoader
from typing import List


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
            content = response.content.replace("\n", " ")
            # Add a space after content
            file.write(f"{content} ")
    return None


def create_db(vertextai_embedding_model, dimension, splits, vector_store):
    if vertextai_embedding_model == "text-embedding-005":
        assert dimension == 768
        vector_store.add_documents(documents=splits)
    return None


