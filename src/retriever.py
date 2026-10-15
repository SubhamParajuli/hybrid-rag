from typing import List,Dict
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

from src.vectorstore import VectorStoreManager

class HybridRetriever:
    def __init__(self,top_k:int=4):
        self.top_k=top_k
        self.vector_manager=VectorStoreManager()
        self.vector_store=self.vector_manager.get_vector_store()

        #setup vector retirever
        self.vector_retriever=self.vector_store.as_retriever(
            search_kwargs={"k":self.top_k}
        )

        # Reconstruct Document objects from vector store for BM25 keyword index
        all_docs=self.vector_store.get()
        documents=[
            Document(page_content=text,metadata=meta)
            for text,meta in zip(all_docs["documents"],all_docs["metadatas"])
        ]

        if not documents:
            raise ValueError("Vector store is empty. Please run ingestion first.")

        self.bm25_retriever=BM25Retriever.from_documents(documents)
        self.bm25_retriever.k=self.top_k


    def reciprocal_rank_fusion(
            self,
            vector_docs:List[Document],
            bm25_docs:List[Document],
            k:int=60
    )-> List[Document]:
        """RRF algorithm to merge and re-rank vector search + keyword search results."""
        doc_scores:Dict[str,float]={}
        doc_map:Dict[str,Document]={}
        def process_docs(docs:List[Document]):
            for rank,doc in enumerate(docs):
                doc_id=doc.page_content
                if doc_id not in doc_map:
                    doc_map[doc_id]=doc
                    doc_scores[doc_id]=0.0
                    doc_scores[doc_id]+=1.0/(rank+k)

        process_docs(vector_docs)
        process_docs(bm25_docs)
        sorted_doc_ids=sorted(doc_scores,key=doc_scores.get,reverse=True)
        return [doc_map[doc_id] for doc_id in sorted_doc_ids[:self.top_k]]

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Executes vector search and BM25 search, returning fused re-ranked results."""
        vector_results = self.vector_retriever.invoke(query)
        bm25_results = self.bm25_retriever.invoke(query)
        return self.reciprocal_rank_fusion(vector_results, bm25_results)