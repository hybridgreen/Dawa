from sentence_transformers import SentenceTransformer
from pathlib import Path
import os
import numpy as np
import re
import json
from lib.medicine_data import load_cached_docs


def clean_for_embedding(text: str) -> str:
    text = re.sub(r"\*+", " ", text)
    text = re.sub(r"\\u[a-z\d]+", " ", text)

    return text


def verify_model():
    sem = SemanticSearch()
    print(f"Model loaded: {sem.model}")
    print(f"Max sequence length: {sem.model.max_seq_length}")
    pass


def verify_embeddings():
    sem = SemanticSearch()

    try:
        documents = load_cached_docs()
    except Exception:
        print("Failed to load documents, exiting.")

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


def split_by_headers(markdown: str):
    header_pattern = r"\*\*\s*\d+\.\d*\s*\*\*\s*\*\*[a-zA-Z\s]+\s*\*\*"
    sub_pattern = r"\*\*\s*\d+\.\d*\s+[a-zA-Z\s]+\s*\*\*"
    combined_pattern = f"{header_pattern}|{sub_pattern}"

    header_positions = []
    for match in re.finditer(combined_pattern, markdown):
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
        cleaned_content = clean_for_embedding(content)

        section_number = header.split("**")[1].strip()
        parts = [p.strip() for p in header.split("**") if p.strip()]
        section_title = parts[-1] if len(parts) > 1 else ""

        if not is_spc_section(section_number, section_title):
            continue

        sections.append(
            {
                "section_number": section_number,
                "section_title": section_title,
                "content": cleaned_content,
            }
        )

    return sections


def is_spc_section(section_number: str, section_title: str) -> bool:
    title_lower = section_title.lower()

    valid_spc_sections = [
        #'1.',
        #'2.',
        "3.",
        "4.1",
        "4.2",
        "4.3",
        "4.4",
        "4.5",
        "4.6",
        "4.7",
        "4.8",
        "4.9",
        "5.1",
        "5.2",
        "5.3",
        "6.1",  #'6.2',
        "6.3",
        "6.4",
        #'6.5', '6.6',
        #'7.', '8.', '9.',# '10.'
    ]

    exclude = [
        "what",
        "What",
        "how to",
        "possible",
        "contents of the pack",
        "instructions on use",
        "information in braille",
    ]

    if section_number in valid_spc_sections:
        if any(pattern in title_lower for pattern in exclude):
            return False
        return True

    if any(pattern in title_lower for pattern in exclude):
        return False

    if section_number not in valid_spc_sections:
        return False

    return True


def semantic_chunking(text: str, max_chunk_size: int, overlap: int):
    text = text.strip()

    if len(text) == 0:
        return []

    parts = re.split(r"(?<=[.!?])\s+", text)

    i = 0
    chunks = []

    while i < len(parts) - overlap:
        chunk = parts[i : max_chunk_size + i]
        s_chunk = " ".join(chunk)

        chunks.append(s_chunk.strip())

        i += max_chunk_size - overlap

    return chunks


class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.docmap = {}
        self.doc_metadata = {}
        self.cache_path = Path(__file__).parent.parent.parent / "cache"
        self.metadata_path = self.cache_path / "medicine_metadata.json"
        self.embeddings_path = (
            self.cache_path / f"chunk_embeddings_{model_name.replace('/', '-')}.npy"
        )
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

        if not self.cache_path.is_dir():
            os.mkdir(self.cache_path)
        with open(self.embeddings_path, "wb") as f:
            np.save(f, self.embbeddings)
        return self.embbeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents

        for doc in self.documents:
            self.docmap[doc["id"]] = doc["content"]

        if self.embeddings_path.exists():
            with open(self.embeddings_path, "rb") as f:
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
        self.chunk_metadata_path = self.cache_path / "chunk_metadata.json"
        self.chunk_embeddings_path = (
            self.cache_path / f"chunk_embeddings_{model_name.replace('/', '-')}.npy"
        )

    def build_chunk_embeddings(self, documents: list[dict]):
        self.documents = documents
        print("Loading, metadata")
        with open(self.metadata_path, "r") as f:
            self.doc_metadata = json.load(f)

        all_chunks = []
        chunk_metadata = []

        for doc in self.documents:
            doc_id: str = doc["id"]

            sections = split_by_headers(doc["content"])

            if len(sections) > 0:
                for section in sections:
                    all_chunks.append(section)
                    metadata = {
                        "doc_id": doc_id,
                        "section": f"{section['section_number']} {section['section_title']}",
                        "chunk_text": section["content"],
                    }
                    chunk_metadata.append(metadata)
            else:
                continue

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata

        if not self.cache_path.is_dir():
            os.mkdir(self.cache_path)

        with open(self.chunk_metadata_path, "w") as f:
            json.dump(chunk_metadata, f, indent=2)

        with open(self.chunk_embeddings_path, "wb") as f:
            np.save(f, self.chunk_embeddings)

        print(f"Created {len(self.chunk_embeddings)}, chunked embeddings")
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents

        for doc in self.documents:
            doc_id = str(doc["id"])
            self.docmap[doc_id] = doc["content"]

        if self.chunk_metadata_path.exists() and self.chunk_embeddings_path.exists():
            print("Loading chunked embeddings")
            with open(self.chunk_metadata_path, "r") as f:
                self.chunk_metadata = json.load(f)

            with open(self.chunk_embeddings_path, "rb") as f:
                self.chunk_embeddings = np.load(f)

            with open(self.metadata_path, "r") as f:
                self.doc_metadata = json.load(f)

            print(f"Loaded {len(self.chunk_embeddings)} chunked embeddings")
            return self.chunk_embeddings

        else:
            print("Building chunked embeddings")
            return self.build_chunk_embeddings(self.documents)

    def filter_chunks(
        self,
        therapeutic_area: str = None,
        active_substance: str = None,
        atc_code: str = None,
    ) -> list[int]:
        filtered_indices = []

        for idx, chunk_meta in enumerate(self.chunk_metadata):
            doc_id = chunk_meta["doc_id"]
            doc_meta = self.doc_metadata.get(doc_id)

            if not doc_meta:
                continue

            if therapeutic_area:
                doc_areas = doc_meta.get("therapeutic_area", [])
                if therapeutic_area.lower() not in [area.lower() for area in doc_areas]:
                    continue

            if active_substance:
                doc_substance = doc_meta.get("active_substance", [])
                if active_substance.lower() not in [
                    substance.lower() for substance in doc_substance
                ]:
                    continue

            if atc_code:
                doc_atc: str = doc_meta.get("atc_code", "")
                if not doc_atc or not doc_atc.startswith(atc_code):
                    continue

            filtered_indices.append(idx)

        return filtered_indices

    def filtered_search(
        self,
        query: str,
        limit: int = 10,
        therapeutic_area: str = None,
        active_substance: str = None,
        atc_code: str = None,
    ):
        if self.chunk_embeddings is None or self.chunk_metadata is None:
            raise ValueError(
                "Missing data. Call `load_or_create_chunk_embeddings` first."
            )

        if not self.doc_metadata:
            with open(self.metadata_path, "r") as f:
                self.doc_metadata = json.load(f)

        chunk_scores = []

        filtered_indices = self.filter_chunks(
            therapeutic_area=therapeutic_area,
            active_substance=active_substance,
            atc_code=atc_code,
        )

        if len(filtered_indices) == 0:
            print("No documents match the filter criteria")
            return []

        print(f"{len(filtered_indices)} filtered docs")

        embedded_q = self.generate_embedding(query)

        for idx in filtered_indices:
            chunk_emb = self.chunk_embeddings[idx]
            doc_id: str = self.chunk_metadata[idx]["doc_id"]

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
                        "name": med["name"],
                        "section": score["metadata"]["section"],
                        "text": score["metadata"]["chunk_text"],
                        "score": round(score["score"], 5),
                    }
                )

        return result
