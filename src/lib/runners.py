import typer
from typing import Annotated
from src.lib.hybrid_search import HybridSearch
from src.lib.medicine_data import process_all_pdfs, load_cached_docs
from src.config import config as cfg


def run_hybrid_search(
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

    try:
        documents = load_cached_docs()
    except Exception:
        print("No document found, rebuilding pdfs.")

    if not documents:
        documents = process_all_pdfs(
            "/Users/yasseryaya-oye/workspace/hybridgreen/dawa/data/pdf"
        )

    search_engine = HybridSearch(documents=documents, model_name=cfg.model)

    print("Starting search")

    results = search_engine.rrf_search(
        query,
        k=60,
        limit=limit,
        therapeutic_area=therapeutic_area,
        active_substance=active_substance,
        atc_code=atc_code,
    )

    return results
