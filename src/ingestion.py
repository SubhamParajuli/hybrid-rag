import os
from pathlib import Path
from typing import List, Any
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader,DirectoryLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import JSONLoader,UnstructuredMarkdownLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from src.config import (
    CHUNK_OVERLAP,CHUNK_SIZE,VECTOR_DB_PATH,EMBEDDING_MODEL_NAME
)

class DocumentIngester:
        def __init__(self,data_dir:str="./data"):
            self.data_dir=data_dir
            self.embeddings=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
            self.text_splitter=RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                add_start_index=True
            )

        def load_documents(self)->List[Document]:
            """Universally loads .txt, .pdf, .md, and .docx files from the data directory 
            and preserves clean source file and article title metadata.
            """
            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir)
                print(f"Created missing data directory at {self.data_dir}. Add some files there!")
                return []

            processed_docs=[]

            for root,_,files in os.walk(self.data_dir):
                for file in files:
                    file_path=os.path.join(root,file)
                    ext=os.path.splitext(file)[1].lower()
                    loader=None

                    try:
                        if ext==".txt":
                            loader=TextLoader(file_path,encoding="utf-8")
                        elif ext==".pdf":
                            loader=PyPDFLoader(file_path)
                        elif ext==".md":
                            laoder=UnstructuredMarkdownLoader(file_path)

                        elif ext in[".docx",".doc"]:
                            loader=Docx2txtLoader(file_path)

                        else:
                            print(f"Skipping unsupported file type : {file}")

                        file_docs=loader.load()

                        for doc in file_docs:
                            doc.metadata["source_file"]=file
                            doc.metadata["article_title"]=os.path.splitext(file)[0].replace("_", " ").title()
                            processed_docs.append(doc)

                    except Exception as e:
                        print(f"Error loading file {file}: {e}")
            print(f"Loaded {len(processed_docs)} total document pages/sections from {self.data_dir}.")
            return processed_docs

        def ingest(self) -> Chroma:
            """Splits documents, attaches chunks, and saves them to the local Chroma vector store."""
            raw_docs = self.load_documents()
            if not raw_docs:
                raise ValueError(f"No documents found in {self.data_dir}. Please add files to proceed.")

            docs = self.text_splitter.split_documents(raw_docs)
            print(f"Split into {len(docs)} chunks.")

            # Create local Chroma vector store
            vector_store = Chroma.from_documents(
                documents=docs,
                embedding=self.embeddings,
                persist_directory=VECTOR_DB_PATH
            )
            print(f"Vector store successfully created and saved at {VECTOR_DB_PATH}.")
            return vector_store


if __name__ == "__main__":
    ingester = DocumentIngester()
    ingester.ingest()