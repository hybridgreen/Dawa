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
import pymupdf


cache_path = Path(__file__).parent.parent.parent / "cache"
embeddings_path = cache_path / "drug_embeddings.npy"
metadata_path = cache_path / "drug_metadata.json"
chunk_metadata_path = cache_path / "chunk_metadata.json"
chunk_embeddings_path = cache_path / "chunk_embeddings.npy"


def verify_model():
    sem = SemanticSearch()
    print(f"Model loaded: {sem.model}")
    print(f"Max sequence length: {sem.model.max_seq_length}")
    pass


def verify_pdf(pdf_path: str) -> bool:
    
    try:
        file_size = Path(pdf_path).stat().st_size
        if file_size < 1024: 
            return False
        
        with open(pdf_path, 'rb') as f:
            header = f.read(4)
            if not header.startswith(b'%PDF'):
                return False
        
        doc = pymupdf.open(pdf_path)
        
        if len(doc) == 0:
            doc.close()
            return False
        
        try:
            first_page = doc[0]
            _ = first_page.get_text("text")
        except Exception:
            doc.close()
            return False
        
        doc.close()
        return True
        
    except Exception as e:
        print(f"PDF verification failed: {e}")
        return False


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
    

def construct_metadata(ema_number, row, updated):

    metadata = {
        "id": str(ema_number),
        "category": str(row["Category"]).lower(),
        "medicine_name": row["Name of medicine"],
        "status": row["Medicine status"].lower(),
        "therapeutic_area": str(row["Therapeutic area (MeSH)"])
        .lower()
        .split(";"),
        "active_substance": str(row["Active substance"]).lower(),
        "atc_code": str(row["ATC code (human)"]).lower(),
        "url": str(row["Medicine URL"]),
        "last_update": str(datetime.strptime(row["Last updated date"], "%d/%m/%Y")),
        "created_at": str(datetime.now()),
        "updated_at": str(datetime.now()) if updated else None,
    }
    
    return metadata


def fetch_documents(med_data_url: str, n_rows:int):
    
    dl_count = 0
    if n_rows == 0:
        n_rows = None
    
    print("Downloading Medicine Data Table")
    med_data_path = fetch_url(med_data_url, "medicine_data_en", "xlsx")
    
    if med_data_path and Path(med_data_path).exists():
        print("Medicine Data Table downloaded.")
        
        data = pd.read_excel(med_data_path, skiprows=8, nrows=n_rows)[
            [
                "Category",
                "Medicine status",
                "Name of medicine",
                "EMA product number",
                "Therapeutic area (MeSH)",
                "Active substance",
                "ATC code (human)",
                "Revision number",
                "Medicine URL",
                "Last updated date",
            ]
        ]
    else:
        raise Exception("Medical Data file download failed)")

    
    if Path(metadata_path).exists():
        try:
            with open(metadata_path, "r") as f:
                doc_metadata: dict = json.load(f)
        except Exception:
            pass
    else:
        doc_metadata = {}
    
    for idx, row in data.iterrows():
        
        medicine_name: str = row["Name of medicine"]
        url_code = str(row["Medicine URL"]).split('/')[-1]
        ema_number: str = row["EMA product number"].split("/")[-1]
        last_update =  datetime.strptime(row["Last updated date"], "%d/%m/%Y")
        metadata = None
        
        if ema_number in doc_metadata:
            metadata = doc_metadata[ema_number]
        
        updated_at = datetime.fromisoformat(metadata['updated_at']) if metadata and metadata.get('updated_at') else None
        
        if not metadata or not updated_at or last_update > updated_at:
            
            try:
                print(f"Downloading data for {medicine_name}, number: {ema_number}...")
                pdf_path = fetch_url(
                    f"https://www.ema.europa.eu/en/documents/product-information/{url_code.lower()}-epar-product-information_en.pdf",
                    f"pdf/{ema_number}-en",
                    "pdf",
                )
                
                if pdf_path :
                    if not verify_pdf(pdf_path):
                        raise Exception("Invalid pdf file - Use download command again") 
                    
                    doc_metadata[str(ema_number)] = construct_metadata(ema_number, row, True)
                    print("Success")
                    dl_count += 1  
                    
            except Exception as e:
                print(f"Error - Downloading {medicine_name}, number: {ema_number}: {str(e)}")
                doc_metadata[ema_number] = construct_metadata(ema_number, row, False)
                continue

    Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(doc_metadata, f, indent=2, default=str)
    
    print(f"Successfully downloaded {dl_count} docs")
    

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

    def filter_chunks(
        self,
        category: str = None,
        status: str = None,
        therapeutic_area: str = None,
        active_substance: str = None,
        atc_code: str = None,
    ) -> list[int]:
    
        filtered_indices = []
        
        for idx, chunk_meta in enumerate(self.chunk_metadata):
            doc_id = chunk_meta['doc_id']
            doc_meta = self.doc_metadata.get(doc_id)
            
            if not doc_meta:
                continue
            
            if category and doc_meta.get('category') != category.lower():
                continue
            
            if status and doc_meta.get('status') != status.lower():
                continue
            
            if therapeutic_area:
                doc_areas = doc_meta.get('therapeutic_area', [])
                if therapeutic_area.lower() not in [area.lower() for area in doc_areas]:
                    continue
            
            if active_substance:
                substance = doc_meta.get('active_substance', '')
                if active_substance.lower() not in substance.lower():
                    continue
            
            if atc_code:
                doc_atc:str = doc_meta.get('atc_code', '')
                if not doc_atc or not doc_atc.startswith(atc_code):
                    continue
                
            filtered_indices.append(idx)
                
        return filtered_indices

    def filtered_search(self, query: str, limit: int = 10,
        category: str = None,
        therapeutic_area: str = None,
        active_substance: str = None,
        atc_code: str = None,
        status: str = None):
        
        if self.chunk_embeddings is None or self.chunk_metadata is None:
            raise ValueError(
                "Missing data. Call `load_or_create_chunk_embeddings` first."
            )

        if not self.doc_metadata:
            with open(metadata_path, "r") as f:
                self.doc_metadata = json.load(f)

        chunk_scores = []
        
        filtered_indices = self.filter_chunks(
            category=category,
            therapeutic_area=therapeutic_area,
            active_substance=active_substance,
            atc_code=atc_code,
            status= status)
        
        if len(filtered_indices):
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
            print(doc_id)
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
                        "section": score["metadata"]["section"],
                        "text": score["metadata"]["chunk_text"],
                        "score": round(score["score"], 5),
                    }
                )

        return result
