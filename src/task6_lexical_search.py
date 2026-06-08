"""
Task 6 - Lexical Search bang BM25.

BM25 cham diem dua tren tan suat tu, do hiem cua tu trong corpus va chuan hoa
do dai document. Corpus dung cung cac chunk cua Task 4 de dense va lexical
retrieval co chung don vi ket qua.
"""

import json
import re
from functools import lru_cache

from rank_bm25 import BM25Okapi

from .task4_chunking_indexing import (
    LOCAL_INDEX_PATH,
    chunk_documents,
    load_documents,
)


def tokenize(text: str) -> list[str]:
    """Tokenize tieng Viet don gian, giu chu va so, bo dau cau."""
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def load_corpus() -> list[dict]:
    """Doc chunks tu local index; neu chua co thi chunk Markdown truc tiep."""
    if LOCAL_INDEX_PATH.exists():
        records = json.loads(LOCAL_INDEX_PATH.read_text(encoding="utf-8"))
        return [
            {
                "content": record["content"],
                "metadata": record.get("metadata", {}),
            }
            for record in records
            if record.get("content")
            and "khong co text layer" not in record["content"].lower()
        ]
    return chunk_documents(load_documents())


def build_bm25_index(corpus: list[dict]) -> BM25Okapi:
    """Xay dung BM25Okapi index tu corpus."""
    tokenized_corpus = [tokenize(doc["content"]) for doc in corpus]
    return BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)


@lru_cache(maxsize=1)
def _cached_index() -> tuple[list[dict], BM25Okapi]:
    corpus = load_corpus()
    return corpus, build_bm25_index(corpus)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Tra ve top_k chunks co BM25 score cao nhat."""
    query_tokens = tokenize(query)
    if not query_tokens or top_k <= 0:
        return []

    corpus, bm25 = _cached_index()
    if not corpus:
        return []

    scores = bm25.get_scores(query_tokens)
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: float(scores[index]),
        reverse=True,
    )

    results = []
    for index in ranked_indices:
        score = float(scores[index])
        if score <= 0:
            continue
        results.append(
            {
                "content": corpus[index]["content"],
                "score": score,
                "metadata": corpus[index].get("metadata", {}),
            }
        )
        if len(results) >= top_k:
            break
    return results


if __name__ == "__main__":
    for result in lexical_search(
        "Dieu 248 tang tru trai phep chat ma tuy",
        top_k=5,
    ):
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
