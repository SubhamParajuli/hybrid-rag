from typing import List,Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma

from src.config import EMBEDDING_MODEL_NAME,VECTOR_DB_PATH


class VectorStoreManager:
    """Encapsulates Hugging Face embedding model initialization and 
    Chroma vector database management in one place.
    """
    def __init__(self):
       #initialize embeddings
       print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
       self.embeddings=HuggingFaceEmbeddings(
           model_name=EMBEDDING_MODEL_NAME,
           model_kwargs={"device":"gpu"},
           encode_kwargs={"normalize_embeddings":True}
       )
       self._vector_store:Optional[Chroma]=None


    def get_vector_store(self)->Chroma:
        """Returns or initializes the local Chroma vector store."""
        if self._vector_store is None:
            self._vector_store=Chroma(
                persist_directory=VECTOR_DB_PATH,
                embedding_function=self.embeddings
            )

        return self._vector_store


    def create_vector_store(self,documents:List[Document])->Chroma:
        """Embeds incoming document chunks and persists them to Chroma DB."""
        print(f"Creating vector store with {len(documents)} document chunks...")
        self._vector_store=Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=VECTOR_DB_PATH
        )
        print(f"Vector store successfully persisted at '{VECTOR_DB_PATH}'.")
        return self._vector_store

    def similarity_search(self,query:str,k:int=4) ->List[Document]:
        """Direct similarity search over vectors."""
        store=self.get_vector_store()
        return store.similarity_search(query,k=k)


if __name__ == "__main__":
    # Standalone test
    vm = VectorStoreManager()
    store = vm.get_vector_store()
    print("VectorStoreManager successfully initialized.")
