from sentence_transformers import SentenceTransformer
from pathlib import Path
from .utils import load_file_data, fetch_url
import os
import numpy as np
import re
import json
import string
import pymupdf4llm
import pandas as pd
from datetime import datetime

cache_path = Path(__file__).parent.parent.parent / "cache"
index_path = cache_path / "index.pkl"
docmap_path = cache_path / "docmap.pkl"
doc_length_path = cache_path / "doc_length.pkl"
embeddings_path = cache_path / "drug_embeddings.npy"
metadata_path = cache_path / "drug_metadata.json"
chunk_metadata_path = cache_path / "chunk_metadata.json"
chunk_embeddings_path = cache_path / "chunk_embeddings.npy"


def verify_model():
    sem = SemanticSearch()
    print(f"Model loaded: {sem.model}")
    print(f"Max sequence length: {sem.model.max_seq_length}")
    pass


def embed_query_text(query: str):
    sem = SemanticSearch()
    embedding = sem.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def verify_embeddings():
    sem = SemanticSearch()

    documents = load_file_data("movies", "json")["movies"]

    embeddings = sem.load_or_create_embeddings(documents)

    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def extract_markdown(pdf_path: str, ema_number: str):
    md_text = pymupdf4llm.to_markdown(pdf_path)

    md_path = Path(__file__).parent.parent.parent.parent / (
        f"./data/markdown/{ema_number}.md"
    )
    Path(md_path).parent.mkdir(parents=True, exist_ok=True)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    return md_text


def save_file(data, filename: str, extension: str):
    filepath = Path(__file__).parent.parent.parent.parent / (
        f"./data/{filename}.{extension}"
    )
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    return filepath


def split_by_headers(markdown: str):
    pattern = r"\*\*\s*\d+\.\d*\s*\*\*\s*\*\*[a-z,A-Z,\s]+\s*\*\*"

    header_positions = []
    for match in re.finditer(pattern, markdown):
        header_text = match.group()
        position = match.start()
        header_positions.append((position, header_text))

    if not header_positions:
        print("No headers found!")
        return []

    sections = []
    for i, (pos, header) in enumerate(header_positions):
        if i + 1 < len(header_positions):
            end_pos = header_positions[i + 1][0]
        else:
            end_pos = len(markdown)

        content = markdown[pos:end_pos].strip()

        section_number = header.split("**")[1].strip()
        parts = [p.strip() for p in header.split("**") if p.strip()]
        section_title = parts[-1] if len(parts) > 1 else ""

        sections.append(
            {
                "section_number": section_number,
                "section_title": section_title,
                "header": header,
                "content": content,
            }
        )

    return sections


def semantic_chunking(text: str, max_chunk_size: int, overlap: int):
    text = text.strip()

    if len(text) == 0:
        return []

    parts = re.split(r"(?<=[.!?])\s+", text)

    if len(parts) == 1 and not parts[0].endswith(string.punctuation):
        return parts

    i = 0
    chunks = []

    while i < len(parts) - overlap:
        chunk = parts[i : max_chunk_size + i]
        s_chunk = " ".join(chunk)

        chunks.append(s_chunk.strip())

        i += max_chunk_size - overlap

    return chunks


def hybrid_chunking(header: str, text: str, max_chunk_size: int, overlap: int):
    text = text.strip()

    if len(text) == 0:
        return []

    parts = re.split(r"(?<=[.!?])\s+", text)

    if len(parts) == 1 and not parts[0].endswith(string.punctuation):
        parts[0] = f"{header}\n\n{parts[0]}"
        return parts

    i = 0
    chunks = [header]

    while i < len(parts) - overlap:
        chunk = parts[i : max_chunk_size + i]
        s_chunk = " ".join(chunk)

        chunks.append(s_chunk.strip())

        i += max_chunk_size - overlap

    return chunks


def fetch_documents(med_data_url: str):
    print("Downloading Medicine Data Table")
    med_data_path = fetch_url(med_data_url, "medicine_data_en", "xlsx")

    if med_data_path:
        print("Medicine Data Table downloaded.")
        data = pd.read_excel(med_data_path, skiprows=8, nrows=100)[
            [
                "Name of medicine",
                "EMA product number",
                "Therapeutic area (MeSH)",
                "Active substance",
                "Revision number",
                "Medicine URL",
            ]
        ]

    paths = []
    doc_metadata = {}
    for idx, row in data.iterrows():
        medicine_name: str = row["Name of medicine"]
        ema_number: str = row["EMA product number"].split("/")[-1]
        try:
            print(f"Downloading data for {medicine_name} , number: {ema_number}...")
            pdf_path = fetch_url(
                f"https://www.ema.europa.eu/en/documents/product-information/{medicine_name.lower()}-epar-product-information_en.pdf",
                f"pdf/{ema_number}-en",
                "pdf",
            )

            if pdf_path:
                print("Success")
                paths.append(pdf_path)
                metadata = {
                    "id": str(ema_number),
                    "medicine_name": row["Name of medicine"],
                    "therapeutic_area": str(row["Therapeutic area (MeSH)"])
                    .lower()
                    .split(";"),
                    "active_substance": str(row["Active substance"]).lower(),
                    "url": str(row["Medicine URL"]),
                    "created_at": str(datetime.now()),
                    "updated_at": str(datetime.now()),
                }
                doc_metadata[str(ema_number)] = metadata
        except Exception as e:
            print(f"Error - Downloading {medicine_name}, number: {ema_number}", str(e))
            metadata = {
                "id": str(ema_number),
                "medicine_name": row["Name of medicine"],
                "therapeutic_area": str(row["Therapeutic area (MeSH)"])
                .lower()
                .split(";"),
                "active_substance": str(row["Active substance"]).lower(),
                "url": str(row["Medicine URL"]),
                "created_at": str(datetime.now()),
                "updated_at": None,
            }
            doc_metadata[str(ema_number)] = metadata
            continue

    Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(doc_metadata, f, indent=2)

    return paths


class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.docmap = {}
        self.doc_metadata = {}
        print("Semantic Search Engine initialised")

    def generate_embedding(self, text: str):
        if text.strip() == "":
            raise ValueError("Empty input")

        encoded = self.model.encode([text])

        return encoded[0]

    def build(self, documents):
        self.documents = documents

        content = []

        for doc in self.documents:
            self.docmap[int(doc["id"])] = doc["content"]
            content.append(f"{doc['content']}")

        self.embbeddings = self.model.encode(content, show_progress_bar=True)

        if not cache_path.is_dir():
            os.mkdir(cache_path)
        with open(embeddings_path, "wb") as f:
            np.save(f, self.embbeddings)
        return self.embbeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents

        for doc in self.documents:
            self.docmap[doc["id"]] = doc["content"]

        if embeddings_path.exists():
            with open(embeddings_path, "rb") as f:
                self.embeddings = np.load(f)
                if len(self.embeddings) == len(documents):
                    return self.embeddings

        return self.build(self.documents)

    def search(self, query, limit):
        if self.embeddings is None or len(self.embeddings) == 0:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        emb_query = self.generate_embedding(query)

        sims = []
        for emb_doc, doc in zip(self.embeddings, self.docmap.values()):
            sims.append((cosine_similarity(emb_query, emb_doc), doc))

        sims.sort(key=lambda x: x[0], reverse=True)

        results = []
        for i in range(min(limit, len(sims))):
            current_doc = sims[i]
            results.append(
                {
                    "score": current_doc[0],
                    "doc": current_doc[1],
                }
            )

        return results


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[dict]):
        self.documents = documents
        print("Loading, metadata")
        with open(metadata_path, "r") as f:
            self.doc_metadata = json.load(f)

        all_chunks = []
        chunk_metadata = []

        for doc in self.documents:
            doc_id: str = doc["id"]
            self.docmap[doc_id] = doc["content"]

            sections = split_by_headers(doc["content"])

            if len(sections) > 0:
                for section in sections:
                    if len(section["content"]) > 20:
                        chunks = semantic_chunking(section["content"], 514, 50)
                        all_chunks.extend(chunks)

                        for chunk in chunks:
                            metadata = {
                                "doc_id": doc_id,
                                "section": section["header"],
                                "chunk_text": chunk,
                            }
                            chunk_metadata.append(metadata)
                    else:
                        continue

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata

        if not cache_path.is_dir():
            os.mkdir(cache_path)

        with open(chunk_metadata_path, "w") as f:
            json.dump(chunk_metadata, f, indent=2)

        with open(chunk_embeddings_path, "wb") as f:
            np.save(f, self.chunk_embeddings)

        print(f"Created {len(self.chunk_embeddings)}, chunked embeddings")
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents

        for doc in self.documents:
            doc_id = str(doc["id"])
            self.docmap[doc_id] = doc["content"]

        if chunk_metadata_path.exists() and chunk_embeddings_path.exists():
            print("Loading chunked embeddings")
            with open(chunk_metadata_path, "r") as f:
                self.chunk_metadata = json.load(f)

            with open(chunk_embeddings_path, "rb") as f:
                self.chunk_embeddings = np.load(f)

            with open(metadata_path, "r") as f:
                self.doc_metadata = json.load(f)

            print(f"Loaded {len(self.chunk_embeddings)} chunked embeddings")
            return self.chunk_embeddings

        else:
            print("Building chunked embeddings")
            return self.build_chunk_embeddings(self.documents)

    def search_chunks(self, query: str, limit: int = 10, therapeutic_area: str = ""):
        if self.chunk_embeddings is None or self.chunk_metadata is None:
            raise ValueError(
                "Missing data. Call `load_or_create_chunk_embeddings` first."
            )

        if not self.doc_metadata:
            with open(metadata_path, "r") as f:
                self.doc_metadata = json.load(f)

        embedded_q = self.generate_embedding(query)

        chunk_scores = []

        for idx, chunk_emb in enumerate(self.chunk_embeddings):
            doc_id: str = self.chunk_metadata[idx]["doc_id"]
            metadata = self.doc_metadata[doc_id]

            print(str(metadata["therapeutic_area"]))
            if (
                therapeutic_area.lower()
                in str(metadata["therapeutic_area"]).lower().split()
            ):
                score = cosine_similarity(embedded_q, chunk_emb)
                chunk_scores.append(
                    {
                        "chunk_idx": idx,
                        "score": score,
                        "metadata": self.chunk_metadata[idx],
                    }
                )

        sorted_scores = sorted(
            chunk_scores, key=lambda item: item["score"], reverse=True
        )

        seen_docs = set()
        top_scores = []

        for data in sorted_scores:
            doc_id = data["metadata"]["doc_id"]

            if doc_id not in seen_docs:
                top_scores.append(data)
                seen_docs.add(doc_id)

                if len(top_scores) >= limit:
                    break

        result = []

        for score in top_scores:
            doc_id = score["metadata"]["doc_id"]
            med = self.doc_metadata.get(doc_id)
            if med:
                result.append(
                    {
                        "id": med["id"],
                        "name": med["medicine_name"],
                        "therapeutic_area": med["therapeutic_area"],
                        "section": score["metadata"]["section"],
                        "text": score["metadata"]["chunk_text"],
                        "score": round(score["score"], 5),
                    }
                )

        return result
