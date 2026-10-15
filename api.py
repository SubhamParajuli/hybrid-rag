import os
from typing import List, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from pydantic import BaseModel

from src.chain import RAGChain
from src.ingestion import DocumentIngester

DATA_DIR = "./data"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Safely loads the RAG chain at startup. If the vector store is empty 
    on a fresh install, it catches the error so FastAPI still starts up 
    successfully, allowing the user to upload documents via the UI.
    """
    print("Server starting up: Initializing RAG components...")
    try:
        app.state.rag_chain = RAGChain()
        print("✅ RAG Chain loaded successfully!")
    except Exception as e:
        print(f"Warning during startup (vector store likely empty): {e}")
        app.state.rag_chain = None  # Will be initialized after first ingestion
    
    yield
    
    app.state.rag_chain = None
    print("🛑 Server shutting down...")


app = FastAPI(
    title="Open-Source Hybrid RAG API",
    description="API for querying documents using Hybrid Search and local Ollama.",
    version="1.0",
    lifespan=lifespan
)


class ChatMessage(BaseModel):
    role: str
    content: str


class QueryRequest(BaseModel):
    question: str
    chat_history: Optional[List[ChatMessage]] = []


class QueryResponse(BaseModel):
    answer: str


@app.post("/ask", response_model=QueryResponse)
def ask_question(request_data: QueryRequest, req: Request):
    """Answers user queries while retaining session history."""
    try:
        rag_chain = req.app.state.rag_chain
        
        if not rag_chain:
            raise HTTPException(
                status_code=400, 
                detail="Vector store is empty. Please upload and ingest documents first."
            )
        
        formatted_history = []
        for msg in request_data.chat_history:
            role = "human" if msg.role == "user" else "ai"
            formatted_history.append((role, msg.content))

        answer = rag_chain.ask(
            question=request_data.question, 
            chat_history=formatted_history
        )
        return {"answer": answer}
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing query: {str(e)}")


@app.post("/upload-and-ingest")
async def upload_and_ingest(request: Request, files: List[UploadFile] = File(...)):
    """Saves uploaded files, triggers ingestion, and hot-reloads the RAG chain into memory."""
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # Save files securely
        for file in files:
            file_path = os.path.join(DATA_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                while content := await file.read(1024 * 1024):  # Read in 1MB chunks
                    buffer.write(content)

        # Run embedding ingestion
        ingester = DocumentIngester(data_dir=DATA_DIR)
        ingester.ingest()
        
        # Hot-reload the RAGChain in app.state so it picks up the new vectors immediately
        request.app.state.rag_chain = RAGChain()
        
        return {
            "status": "success", 
            "message": f"Successfully uploaded, indexed, and loaded {len(files)} file(s)."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")