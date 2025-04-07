from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from pathlib import Path
import asyncio
from langchain_core.documents import Document
from typing import List


class PDFProcessor:
    def __init__(self, data_directory: Path):
        self.data_directory = data_directory


    async def load_pdf(self, pdf_path: Path):
        loader = PyPDFLoader(str(pdf_path))
        pages = []
        async for page in loader.alazy_load():
            pages.append(page)
        return pages
    

    def save_to_txt(self, pages: List[Document], txt_path: Path):
        lines = [page.page_content for page in pages]
        content = "\n".join(lines)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(content)


    async def process_pdf(self, pdf_path: Path):
        txt_path = pdf_path.with_suffix(".txt")
        pages = await self.load_pdf(pdf_path)
        self.save_to_txt(pages, txt_path)


    async def process_pdfs_in_directory(self):
        pdf_paths = list(self.data_directory.glob("*.pdf"))  
        tasks = [self.process_pdf(pdf) for pdf in pdf_paths]
        await asyncio.gather(*tasks)


    def load_txts_in_directory(self):
        loader = DirectoryLoader(
            str(self.data_directory), 
            glob="*.txt", 
            loader_cls=TextLoader, 
            use_multithreading=True
        )
        return loader.load()
    

