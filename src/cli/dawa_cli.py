import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import typer
from lib.runners import run_hybrid_search
from typing import Annotated
from lib.semantic_search import ChunkedSemanticSearch, verify_model, split_by_headers
from lib.hybrid_search import HybridSearch
from lib.medicine_data import (
    download_med_data,
    download_pdfs,
    process_all_pdfs,
    load_cached_docs,
)

from lib.utils import tokenise_string
from lib.gemini import gemini_ai
from config import config as cfg

app = typer.Typer(help="Semantic Search CLI")
gemini = gemini_ai("gemini-2.5-flash")

@app.command()
def verify():
    """Verify the model"""
    verify_model()

@app.command()
def tokenise(text: str):
    print(f"Input text: {text}")
    output = tokenise_string(text)
    print("Tokens produced:")
    for token in output:
        print(token)


@app.command()
def download(n_rows: Annotated[int, typer.Argument()] = 0):
    """Download medical pdf data from the EMA Website - Only available in the EU/UK"""
    
    med_data_path = download_med_data(cfg.med_data_url)

    download_pdfs(med_data_path, n_rows)


@app.command()
def build_documents(rebuild: Annotated[bool, typer.Option("--rebuild")] = False):
    process_all_pdfs(
        "/Users/yasseryaya-oye/workspace/hybridgreen/dawa/data/pdf", rebuild=rebuild
    )


@app.command()
def build_embeddings():
    """Builds embeddings for all pdfs found in the download folder
    """
    documents = load_cached_docs()

    if not documents:
        documents = process_all_pdfs(
            "/Users/yasseryaya-oye/workspace/hybridgreen/dawa/data/pdf", rebuild=True
        )

    search = HybridSearch(documents= documents, model_name=cfg.model)
    search.build_chunk_embeddings(documents)
    search.build_index()


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="Search query")],
    limit: Annotated[int, typer.Option("--limit", "-l", help="Number of results")] = 5,
    therapeutic_area: Annotated[
        str, typer.Option("--therapeutic-area", "-t", help="Filter by therapeutic area")
    ] = None,
    active_substance: Annotated[
        str, typer.Option("--active-substance", "-s", help="Filter by active substance")
    ] = None,
    atc_code: Annotated[str, typer.Option("--atc", help="Filter by ATC code")] = None,
):
    query = gemini.spell(query)

    results = run_hybrid_search(
        query=query,
        limit=limit,
        therapeutic_area=therapeutic_area,
        active_substance=active_substance,
        atc_code=atc_code
    )

    response = gemini.question(query, results)

    print(response)


if __name__ == "__main__":
    app()
