import typer
from typing import Annotated
from lib.gemini import gemini_ai

from lib.semantic_search import (
    ChunkedSemanticSearch,
    verify_model,
    split_by_headers,
    verify_embeddings
)
from lib.medicine_data import (
    download_med_data,
    download_pdfs,
    process_all_pdfs,
    pdf_to_md,
    load_cached_docs)

from lib.hybrid_search import HybridSearch

app = typer.Typer(help="Semantic Search CLI")


@app.command()
def verify():
    """Verify the model"""
    verify_model()

@app.command()
def verify_embeddings():
    """Verify the existing embeddings
    Available models
    'all-MiniLM-L6-v2'- 384 dims
    
    'sentence-transformers/all-mpnet-base-v2' - 768 dims, better quality

    'BAAI/bge-large-en-v1.5' - 1024 dims, medical-focused
    
    'pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb' 1024 dims, specifically for biomedical
     """ 
    
    verify_embeddings()


@app.command()
def split_pdf(filepath: str):
    markdown = pdf_to_md(filepath)
    print(f"Extracted {len(markdown)} characters")

    sections = split_by_headers(markdown)
    print(f"Found {len(sections)} sections")

    for section in sections:
        print(section["section_number"], section["section_title"])


@app.command()
def download(med_data_url: str, n_rows: Annotated[int, typer.Argument()] = 0):
    """Download medical pdf data from the EMA Website - Only available in the EU/UK"""
    med_data_path = download_med_data(med_data_url)
    
    download_pdfs(med_data_path, n_rows)

@app.command()
def build_documents(rebuild: bool = True ):
    process_all_pdfs(
        "/Users/yasseryaya-oye/workspace/hybridgreen/dawa/data/pdf",
        rebuild = rebuild
        )

@app.command()
def build_embeddings(model: Annotated[str, typer.Argument(help="Embedding model")] = 'all-MiniLM-L6-v2' ):
    """Builds embeddings for all pdfs found in the download folder
    
    Available models
    'all-MiniLM-L6-v2'- 384 dims
    
    'all-mpnet-base-v2' - 768 dims, better quality

    'BAAI/bge-large-en-v1.5' - 1024 dims, medical-focused
    
    'pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb' 1024 dims, specifically for biomedical
    
    """
    documents = load_cached_docs()
    
    if not documents:
        documents = process_all_pdfs(
            "/Users/yasseryaya-oye/workspace/hybridgreen/dawa/data/pdf",
            rebuild= True
        )
    
    sem = ChunkedSemanticSearch(model)
    sem.build_chunk_embeddings(documents)


@app.command()
def query(
    query: Annotated[str, typer.Argument(help="Search query")],
    limit: Annotated[int, typer.Option("--limit", "-l" ,help="Number of results")] = 5,
    therapeutic_area: Annotated[str, typer.Option("--therapeutic-area", "-t", help="Filter by therapeutic area")] = None,
    active_substance: Annotated[str, typer.Option("--active-substance", "-a", help="Filter by active substance")] = None,
    atc_code: Annotated[str, typer.Option("--atc", help="Filter by ATC code")] = None,
):
    """Search the datasets for relevant matches, uses AI to enhance the search and answer the question"""

    try:
        documents = load_cached_docs()
    except Exception:
        print("No document found, rebuilding pdfs.")

    if not documents:
        documents = process_all_pdfs(
            "/Users/yasseryaya-oye/workspace/hybridgreen/dawa/data/pdf"
        )
        
    #Current model , 384 dims
    model = 'all-MiniLM-L6-v2'
    
    # 768 dims, better quality
    #model = 'sentence-transformers/all-mpnet-base-v2'

    # 1024 dims, medical-focused
    #model = 'BAAI/bge-large-en-v1.5'

    # 1024 dims, specifically for biomedical
    #model = 'pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb'
    
    sem = ChunkedSemanticSearch(model)
    
    sem.load_or_create_chunk_embeddings(documents)

    result = sem.filtered_search(
        query=query,
        limit=limit,
        therapeutic_area=therapeutic_area,
        active_substance=active_substance,
        atc_code=atc_code
    )
    
    for idx, res in enumerate(result,1):
        
        print(f"{idx}. Medicine name: {res['name']}")
        print(f"    Retrieved section: {res['section']}")
        print(f"    Score: {res['score']}")
        print(f"    Content: {res['text'][:100]}")

@app.command()
def hybrid(
    query: Annotated[str, typer.Argument(help="Search query")],
    limit: Annotated[int, typer.Option("--limit", "-l" ,help="Number of results")] = 5,
    therapeutic_area: Annotated[str, typer.Option("--therapeutic-area", "-t", help="Filter by therapeutic area")] = None,
    active_substance: Annotated[str, typer.Option("--active-substance", "-a", help="Filter by active substance")] = None,
    atc_code: Annotated[str, typer.Option("--atc", help="Filter by ATC code")] = None,
):
    
    try:
        documents = load_cached_docs()
    except Exception:
        print("No document found, rebuilding pdfs.")

    if not documents:
        documents = process_all_pdfs(
            "/Users/yasseryaya-oye/workspace/hybridgreen/dawa/data/pdf"
        )
        
    #Current model , 384 dims
    #model = 'all-MiniLM-L6-v2'
    
    # 768 dims, better quality
    #model = 'sentence-transformers/all-mpnet-base-v2'

    # 1024 dims, medical-focused
    model = 'BAAI/bge-large-en-v1.5'

    # 1024 dims, specifically for biomedical
    #model = 'pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb'
    
    search_engine = HybridSearch(documents= documents, model_name=model)
    search_engine.load_or_create_chunk_embeddings(documents)
    
    results = search_engine.filtered_weighted_search(query,
        alpha= 0.5,
        limit=limit,
        therapeutic_area=therapeutic_area,
        active_substance=active_substance,
        atc_code=atc_code
        )
    
    for idx, result in enumerate(results,1):
        res = result[1]
        print(f"{idx}. Medicine name: {res['name']}")
        print(f"    Retrieved section: {res['section']}")
        print(f"    Content: {res['text'][:100]}")
        print(f"    Scores: BM25:{res['BM25'] }, SEM: {res['SEM']}, Hybrid: {[res['HYB']]}")
    

@app.command()
def question(
    query: Annotated[str, typer.Argument(help="Search query")],
    limit: Annotated[int, typer.Option("--limit", "-l" ,help="Number of results")] = 5,
    category: Annotated[str, typer.Option("--category", "-c", help="Filter by category")] = "human",
    therapeutic_area: Annotated[str, typer.Option("--therapeutic-area", "-t", help="Filter by therapeutic area")] = None,
    active_substance: Annotated[str, typer.Option("--active-substance", "-a", help="Filter by active substance")] = None,
    atc_code: Annotated[str, typer.Option("--atc", help="Filter by ATC code")] = None,
    status: Annotated[str, typer.Option(help="Filter by status")] = None,
):
    """Search the datasets for relevant matches, uses AI to enhance the search and answer the question"""

    gemini = gemini_ai("gemini-2.5-flash")

    query = gemini.spell(query)

    try:
        documents = load_cached_docs()
    except Exception:
        print("No document found, rebuilding pdfs.")

    if not documents:
        documents = process_all_pdfs(
            "/Users/yasseryaya-oye/workspace/hybridgreen/dawa/data/pdf"
        )
        
    #Current 384
    model = 'all-MiniLM-L6-v2'
    
    # 768 dims, better quality
    #model = 'sentence-transformers/all-mpnet-base-v2'

    # 1024 dims, medical-focused
    #model = 'BAAI/bge-large-en-v1.5'

    # 1024 dims, specifically for biomedical
    #model = 'pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb'
    
    sem = ChunkedSemanticSearch(model)
    
    sem.load_or_create_chunk_embeddings(documents)

    result = sem.filtered_search(
        query=query,
        limit=limit,
        category=category,
        therapeutic_area=therapeutic_area,
        active_substance=active_substance,
        atc_code=atc_code,
        status=status
    )

    ai_response = gemini.question(query, result)

    print(ai_response)


if __name__ == "__main__":
    app()
