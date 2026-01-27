import typer
from typing import Annotated
from lib.gemini import gemini_ai
from lib.semantic_search import (
    ChunkedSemanticSearch,
    verify_model,
    split_by_headers,
)
from lib.medicine_data import download_med_data, download_pdfs, process_all_pdfs, pdf_to_md
from lib.utils import load_cached_docs

app = typer.Typer(help="Semantic Search CLI")


@app.command()
def verify():
    """Verify the model"""
    verify_model()


@app.command()
def download(med_data_url: str, n_rows: Annotated[int, typer.Argument()] = 0):
    """Download medical pdf data from the EMA Website - Only available in the EU/UK"""
    med_data_path = download_med_data(med_data_url)
    
    download_pdfs(med_data_path, n_rows)


@app.command()
def split_sections(filepath: str):
    markdown = pdf_to_md(filepath)
    print(f"Extracted {len(markdown)} characters")

    sections = split_by_headers(markdown)
    print(f"Found {len(sections)} sections")

    for section in sections:
        print(section["section_number"], section["section_title"])


@app.command()
def build():
    """Builds embeddings for all pdfs found in the download folder"""

    documents = process_all_pdfs(
        "/Users/yasseryaya-oye/workspace/hybridgreen/dawa/data/pdf"
    )
    sem = ChunkedSemanticSearch()
    sem.load_or_create_chunk_embeddings(documents)


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

    sem = ChunkedSemanticSearch()
    
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
