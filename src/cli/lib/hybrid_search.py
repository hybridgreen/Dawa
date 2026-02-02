from rank_bm25 import BM25Okapi
from .semantic_search import ChunkedSemanticSearch, cosine_similarity
from .utils import normalise_score, tokenise_string
from typing import DefaultDict
import pickle
from pathlib import Path


def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score


def rrf_score(rank, k=60):
    return 1 / (k + rank)


class HybridSearch(ChunkedSemanticSearch):
    def __init__(self, documents, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__(model_name)

        self.index_path = self.cache_path / "bm25_index.pkl"
        self.load_or_create_chunk_embeddings(documents)
        self.bm25 = None
        self.load_index()

    def build_index(self):
        tokenized_texts = [
            tokenise_string(chunk["chunk_text"]) for chunk in self.chunk_metadata
        ]
        self.bm25 = BM25Okapi(tokenized_texts)

        print("✓ BM25 index built")

        try:
            Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)

            with open(self.index_path, "wb") as f:
                pickle.dump(self.bm25, f, protocol=pickle.HIGHEST_PROTOCOL)

            print(f"✓ BM25 index saved to {self.index_path}")
        except Exception as e:
            print(f"Failed to save BM25 Index: {e}")

    def load_index(self):
        
        if Path(self.index_path).exists():
            try:
                with open(self.index_path, "rb") as f:
                    self.bm25 = pickle.load(f)

                print(f"✓ BM25 index loaded from {self.index_path}")
            except Exception as e:
                print(f"Failed to load BM25 Index: {e}")
        else:
            self.build_index()

    def filtered_weighted_search(
        self,
        query,
        alpha=0.5,
        limit=5,
        therapeutic_area: str = None,
        active_substance: str = None,
        atc_code: str = None,
    ):
        if therapeutic_area or active_substance or atc_code:
            filtered_indices = self.filter_chunks(
                therapeutic_area=therapeutic_area,
                active_substance=active_substance,
                atc_code=atc_code,
            )
            if not filtered_indices:
                print("No chunks match filters")
                return []
            print(f"Filtered to {len(filtered_indices)} chunks")
        else:
            filtered_indices = list(range(len(self.chunk_metadata)))

        # Get Normalised BM25 scores
        tokenised_query = tokenise_string(query)
        print("Tokenised query:", tokenised_query)

        bm25_scores_all = self.bm25.get_scores(tokenised_query)
        bm25_filtered = {idx: bm25_scores_all[idx] for idx in filtered_indices}

        min_bm25, max_bm25 = (
            min(list(bm25_filtered.values())),
            max(list(bm25_filtered.values())),
        )

        bm25_norm = {
            idx: normalise_score(score, min_bm25, max_bm25)
            for idx, score in bm25_filtered.items()
        }

        # Get Normalised Semantic scores
        embedded_query = self.model.encode(query)

        sem_scores = {
            idx: cosine_similarity(embedded_query, self.chunk_embeddings[idx])
            for idx in filtered_indices
        }

        min_sem, max_sem = (
            min(list(sem_scores.values())),
            max(list(sem_scores.values())),
        )

        sem_norm = {
            idx: normalise_score(score, min_sem, max_sem)
            for idx, score in sem_scores.items()
        }

        results = {}

        for idx in filtered_indices:
            bm25_score = bm25_norm[idx]
            sem_score = sem_norm[idx]
            chunk = self.chunk_metadata[idx]
            doc_id = self.chunk_metadata[idx].get("doc_id", "")
            med = self.doc_metadata[doc_id]

            results[idx] = {
                "id": doc_id,
                "name": med["name"],
                "section": chunk["section"],
                "text": chunk["chunk_text"],
                "BM25": bm25_score,
                "SEM": sem_score,
                "HYB": hybrid_score(bm25_score, sem_score, alpha),
            }

        sorted_docs = sorted(
            results.items(), key=lambda item: item[1]["HYB"], reverse=True
        )

        return sorted_docs[:limit]

    def rrf_search(
        self,
        query,
        k=0.5,
        limit=5,
        therapeutic_area: str = None,
        active_substance: str = None,
        atc_code: str = None,
    ):
        results = DefaultDict(lambda: {"BM25": 0, "SEM": 0, "RRF": 0})

        if therapeutic_area or active_substance or atc_code:
            filtered_indices = self.filter_chunks(
                therapeutic_area=therapeutic_area,
                active_substance=active_substance,
                atc_code=atc_code,
            )
            if not filtered_indices:
                print("No chunks match filters")
                return []
            print(f"Filtered to {len(filtered_indices)} chunks")
        else:
            filtered_indices = list(range(len(self.chunk_metadata)))

        tokenised_query = tokenise_string(query)
        print("Tokenised query:", tokenised_query)

        bm25_scores_all = self.bm25.get_scores(tokenised_query)
        bm25_filtered = {idx: bm25_scores_all[idx] for idx in filtered_indices}

        bm25_sorted = sorted(
            bm25_filtered.items(), key=lambda item: item[1], reverse=True
        )

        for rank, (idx, score) in enumerate(bm25_sorted, 1):
            results[idx]["BM25"] = rank
            results[idx]['BM25_score'] = score

        embedded_query = self.model.encode(query)

        sem_scores = {
            idx: cosine_similarity(embedded_query, self.chunk_embeddings[idx])
            for idx in filtered_indices
        }

        sem_sorted = sorted(sem_scores.items(), key=lambda item: item[1], reverse=True)

        for rank, (idx, score) in enumerate(sem_sorted, 1):
            results[idx]["SEM"] = rank
            results[idx]['SEM_score'] = score

        output = {}
        for idx in results:
            chunk = self.chunk_metadata[idx]
            doc_id = chunk.get("doc_id", "")
            med = self.doc_metadata[doc_id]
            
            rrf_rank = 0
            rrf_rank += rrf_score(results[idx]["BM25"], k)
            rrf_rank += rrf_score(results[idx]["SEM"], k)
            sem_score = results[idx]["SEM_score"]
            bm_score = results[idx]["BM25_score"]

            output[idx] = {
                "id": doc_id,
                "name": med["name"],
                "section": chunk["section"],
                "text": chunk["chunk_text"],
                "url_code": med.get('url_code', ' '),
                "SEM": sem_score,
                "BM25": bm_score,
                "RRF": rrf_rank
            }

        sorted_docs = sorted(
            output.items(), key=lambda item: item[1]["RRF"], reverse=True
        )
        
        sorted_docs = [doc[1] for doc in sorted_docs]

        return sorted_docs[:limit]
