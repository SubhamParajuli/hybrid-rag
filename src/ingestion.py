import os
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader

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
        """Simple loader supporting standard .txt and .pdf files."""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"Created directory {self.data_dir}. Place your document files here.")
            return []

        processed_docs = []

        for file in os.listdir(self.data_dir):
            file_path = os.path.join(self.data_dir, file)
            if os.path.isdir(file_path):
                continue

            ext = os.path.splitext(file)[1].lower()

            try:
                if ext in [".txt", ".md"]:
                    # Treat markdown and text files simply as plain text
                    loader = TextLoader(file_path, encoding="utf-8")
                elif ext == ".pdf":
                    loader = PyPDFLoader(file_path)
                else:
                    print(f"Skipping unsupported file type: {file}")
                    continue

                file_docs = loader.load()

                # Attach clean metadata
                for doc in file_docs:
                    doc.metadata["source_file"] = file
                    doc.metadata["article_title"] = os.path.splitext(file)[0].replace("_", " ").title()
                    processed_docs.append(doc)

                print(f"Successfully loaded: {file}")

            except Exception as e:
                print(f"Error loading {file}: {e}")

        return processed_docs

    def ingest(self):
        """Loads docs, splits them into chunks, and saves to Chroma."""
        raw_docs = self.load_documents()
        if not raw_docs:
            raise ValueError(f"No documents successfully loaded from {self.data_dir}.")

        chunks = self.text_splitter.split_documents(raw_docs)
        print(f"Split {len(raw_docs)} source pages into {len(chunks)} chunks.")

        # Save to vector store
        self.vector_manager.create_vector_store(chunks)
        print("Ingestion complete and vector store saved!")


if __name__ == "__main__":
    ingester = DocumentIngester()
    ingester.ingest()