import os
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_SIZE, CHUNK_OVERLAP
from src.vectorstore import VectorStoreManager


class DocumentIngester:
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            add_start_index=True
        )
        self.vector_manager = VectorStoreManager()

    def load_documents(self) -> List[Document]:
        """Universal file loader supporting .txt, .pdf, .md, and .docx."""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"Created directory {self.data_dir}. Place your document files here.")
            return []

        processed_docs = []

        for root, _, files in os.walk(self.data_dir):
            for file in files:
                file_path = os.path.join(root, file)
                ext = os.path.splitext(file)[1].lower()

                try:
                    if ext == ".txt":
                        from langchain_community.document_loaders import TextLoader
                        loader = TextLoader(file_path, encoding="utf-8")
                    elif ext == ".pdf":
                        from langchain_community.document_loaders import PyPDFLoader
                        loader = PyPDFLoader(file_path)
                    elif ext == ".md":
                        from langchain_community.document_loaders import UnstructuredMarkdownLoader
                        loader = UnstructuredMarkdownLoader(file_path)
                    elif ext in [".docx", ".doc"]:
                        from langchain_community.document_loaders import Docx2txtLoader
                        loader = Docx2txtLoader(file_path)
                    else:
                        continue

                    file_docs = loader.load()

                    # Attach explicit metadata to each chunk/page
                    for doc in file_docs:
                        doc.metadata["source_file"] = file
                        doc.metadata["article_title"] = os.path.splitext(file)[0].replace("_", " ").title()
                        processed_docs.append(doc)

                except Exception as e:
                    print(f"Error loading {file}: {e}")

        return processed_docs

    def ingest(self):
        """Loads docs, splits them into chunks, and persists via VectorStoreManager."""
        raw_docs = self.load_documents()
        if not raw_docs:
            raise ValueError(f"No documents found in {self.data_dir}.")

        chunks = self.text_splitter.split_documents(raw_docs)
        print(f"Split {len(raw_docs)} files into {len(chunks)} chunks.")

        # Single call to store and persist
        self.vector_manager.create_vector_store(chunks)


if __name__ == "__main__":
    ingester = DocumentIngester()
    ingester.ingest()