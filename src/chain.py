from langchain_core .prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from src.config import OLLAMA_MODEL
from src.retriever import HybridRetriever
from typing import List,Tuple

def format_docs(docs):
    """Turns list of Document objects into a single string with source tags."""
    formatted=[]
    for doc in docs:
        source=doc.metadata.get("source_file","Unknown")
        title=doc.metadata.get("article_title","Untitled")
        formatted.append(f"\n{doc.page_content}")
    return "\n\n".join(formatted)

class RAGChain:
    def __init__(self,top_k:int=4):
        self.retriever=HybridRetriever(top_k=top_k)
        self.llm=ChatOllama(
            model=OLLAMA_MODEL,
            temperature=0.1
        )
        # Prompt supporting context, system instructions, and chat history
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system", 
                "You are a helpful assistant. Answer the user's question based strictly on the provided context below. "
                "If the context does not contain enough info, state that you cannot answer from the documents. "
                "Always cite the source files and titles used."
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "Context:\n{context}\n\nQuestion: {question}")
        ])

    def ask(self, question: str, chat_history: List[Tuple[str, str]] = None) -> str:
        """Executes RAG chain with persistent chat memory."""
        if chat_history is None:
            chat_history = []

        # 1. Retrieve relevant docs based on current question
        docs = self.retriever.get_relevant_documents(question)
        context_str = format_docs(docs)
        
        # 2. Build and run chain
        chain = self.prompt | self.llm | StrOutputParser()
        
        return chain.invoke({
            "context": context_str,
            "question": question,
            "chat_history": chat_history
        })
    
if __name__ == "__main__":
    rag = RAGChain()
    print(rag.ask("What is this project about?"))