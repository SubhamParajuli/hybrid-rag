import os 
from typing import List,Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI,HTTPException,UploadFile,File,Request
from pydantic import BaseModel

from src.chain import RAGChain
from src.ingestion import DocumentIngester

DATA_DIR="./data"

@asynccontextmanager
async def lifespan(app:FastAPI):
   """
    Async context manager: Code before 'yield' runs ONCE at startup.
    Loads models into memory once and attaches to app.state.
    """
   print("Server Startign up: Loading RAG models into memory...") 
   app.state.rag_chain=RAGChain()
   print("RAG Chain & Models loaded successfully!")

   yield

   print("Server shutting down: Cleaning up resources...")
   app.state.rag_chain=None



app = FastAPI(
    title="Open-Source Hybrid RAG API",
    description="API for querying documents using Hybrid Search (Chroma DB + BM25) and local Ollama.",
    version="1.0",
    lifespan=lifespan
)

# --- Data Models ---
class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class QueryRequest(BaseModel):
    question:str
    chat_history:Optional[List[ChatMessage]]=[]

class QueryResponse(BaseModel):
    answer:str


@app.post('/ask',response_model=QueryResponse)
async def ask_question(request_data:QueryRequest, req:Request):
    """Answers user queries while retaining multi-turn session history."""
    try:
       #Access pre-loaded RAG model from app.state(loaded once during startup)
        rag_chain:RAGChain =req.app.state.rag_chain
        formatted_history=[]
        for msg in request_data.chat_history:
            role="human" if msg.role=="user" else "ai"
            formatted_history.append((role,msg.content))

        answer=rag_chain.ask(
            question=request_data.question,
            chat_history=formatted_history
        )
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500,detail=f"Error executing query; {str(e)}")


async def upload_and_ingest(files: List[UploadFile] = File(...)):
    """Uploads documents directly, saves to ./data, and re-indexes vector store."""
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # Save uploaded files locally
        for file in files:
            file_path = os.path.join(DATA_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                buffer.write(await file.read())

        # Trigger Ingestion
        ingester = DocumentIngester(data_dir=DATA_DIR)
        ingester.ingest()
        
        return {
            "status": "success", 
            "message": f"Successfully uploaded and indexed {len(files)} file(s)."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")