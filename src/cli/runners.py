import typer
from typing import Annotated
from lib.hybrid_search import HybridSearch

from lib.medicine_data import process_all_pdfs, load_cached_docs


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
    # Starter model, 384 dims
    model = "all-MiniLM-L6-v2"

    # 768 dims, better quality
    # model = 'sentence-transformers/all-mpnet-base-v2'

    # 1024 dims, medical-focused
    # model = 'BAAI/bge-large-en-v1.5'

    # 1024 dims, specifically for biomedical
    # model = 'pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb'

    try:
        documents = load_cached_docs()
    except Exception:
        print("No document found, rebuilding pdfs.")

    if not documents:
        documents = process_all_pdfs(
            "/Users/yasseryaya-oye/workspace/hybridgreen/dawa/data/pdf"
        )

    search_engine = HybridSearch(documents=documents, model_name=model)
    search_engine.load_index()

    print("Starting search")

    # results = search_engine.filtered_weighted_search(query,
    #    alpha= 0.5,
    #    limit=limit,
    #    therapeutic_area=therapeutic_area,
    #    active_substance=active_substance,
    #    atc_code=atc_code
    #    )

    results = search_engine.rrf_search(
        query,
        k=60,
        limit=limit,
        therapeutic_area=therapeutic_area,
        active_substance=active_substance,
        atc_code=atc_code,
    )

    return results
