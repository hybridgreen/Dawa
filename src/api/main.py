import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from src.lib.gemini import gemini_ai
from src.config import config as cfg
from src.lib.hybrid_search import HybridSearch
from src.lib.medicine_data import process_all_pdfs, load_cached_docs

@asynccontextmanager
async def lifespan(app: FastAPI):
    
    try:
        documents = load_cached_docs()
    except Exception:
        print("No document found, rebuilding pdfs.")

    if not documents:
        documents = process_all_pdfs(
            "/Users/yasseryaya-oye/workspace/hybridgreen/dawa/data/pdf"
        )
    
    print("Loading search engine...")
    
    app.state.search_engine = HybridSearch(documents=documents, model_name=cfg.model)
    
    print("✓ Ready")
    
    yield 
    
    print("Shutting down...")


app = FastAPI(lifespan=lifespan)

origins = [
    "http://localhost",      
    "http://localhost:8080",
    "https://dawa-one.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

gemini = gemini_ai("gemini-2.5-flash")


class ChatForm(BaseModel):
    query: str
    limit: int = 5
    therapeutic_area: str | None = None
    active_substance: str | None = None
    atc_code: str | None = None
    
    
@app.get("/")
def root():
    return {"message":"Welcome to Dawa"}

@app.get("/healthz/")
def health_check():
    if not app.state.search_engine:
        raise HTTPException(status_code= 500, detail= "Unable to initialise search engine") 
    
    return "Healthy"

@app.post("/chats/")
def message( data: ChatForm
):
    if data.query.strip() == "":
        raise HTTPException(status_code=400, detail= "Empty query, please type an question")
    
    query = gemini.spell(data.query)
    
    print("Starting search")

    results = app.state.search_engine.rrf_search(
        query,
        k=60,
        limit=5
    )
    
    response = gemini.question(query, results)

    print(response)
    return(response)