from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel
from src.lib.gemini import gemini_ai
from src.config import config as cfg
from src.lib.hybrid_search import HybridSearch
from src.lib.medicine_data import process_all_pdfs, load_cached_docs

class ChatForm(BaseModel):
    query: str
    limit: int = 5
    therapeutic_area: str | None = None
    active_substance: str | None = None
    atc_code: str | None = None
    
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

gemini = gemini_ai("gemini-2.5-flash")


@app.get("/")
def root():
    return {"message":"Welcome to Dawa"}

@app.post("/chats/")
def message( data: ChatForm
):
    
    query = gemini.spell(data.query)
    
    print("Starting search")

    results = app.state.search_engine.rrf_search(
        query,
        k=60,
        limit=5
    )

    print("Retrieved documents:", results)
    
    response = gemini.question(query, results)

    print(response)
    return(response)