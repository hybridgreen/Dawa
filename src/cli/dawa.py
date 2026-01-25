import typer
from typing import Annotated
from lib.semantic_search import (
    ChunkedSemanticSearch,
    verify_model,
    embed_query_text,
    fetch_documents,
    split_by_headers,
)
from lib.gemini import gemini_ai
from lib.utils import process_all_pdfs, load_cached_docs, pdf_to_md, refresh_documents

app = typer.Typer(help="Semantic Search CLI")


@app.command()
def verify():
    """Verify the model"""
    verify_model()


@app.command()
def embedquery(text: Annotated[str, typer.Argument(help="Text to be processed")]):
    """Checks embedding is working properly"""
    embed_query_text(text)


@app.command()
def download_med_data(med_data_url: str, n_rows: Annotated[int, typer.Argument()] = 10):
    """Download medical pdf data from the EMA Website - Only available in the EU/UK"""
    fetch_documents(med_data_url, n_rows)

@app.command()
def refresh():
    refresh_documents()

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
def query(query: str, limit: int):
    """Search the datasets for relevant matches"""

    gemini = gemini_ai("gemini-2.5-flash")

    query = gemini.spell(query)

    try:
        documents = load_cached_docs()
    except Exception:
        print("No document found, rebuilding pdfs. ")

    if not documents:
        documents = process_all_pdfs(
            "/Users/yasseryaya-oye/workspace/hybridgreen/dawa/data/pdf"
        )

    sem = ChunkedSemanticSearch()
    sem.load_or_create_chunk_embeddings(documents)

    result = sem.search_chunks(query, limit)

    for i, res in enumerate(result):
        print(f"Drug name: {res['name']}")
        print(f"Section : {res['section']}")


@app.command()
def question(
    query: str, limit: int, therapeutic_filter: Annotated[str, typer.Argument()] = ""
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

    result = sem.search_chunks(query, limit, therapeutic_area=therapeutic_filter)

    ai_response = gemini.question(query, result)

    print(ai_response)


if __name__ == "__main__":
    app()
