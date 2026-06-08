"""Task 9 - Hybrid retrieval: dense + BM25 + RRF + Jina + PageIndex."""

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search

SCORE_THRESHOLD = 0.3
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """Chay hybrid retrieval va fallback PageIndex khi evidence yeu."""
    query = query.strip()
    if not query or top_k <= 0:
        return []

    candidate_k = max(top_k * 2, 10)
    dense_results = semantic_search(query, top_k=candidate_k)
    sparse_results = lexical_search(query, top_k=candidate_k)
    merged = rerank_rrf(
        [dense_results, sparse_results],
        top_k=candidate_k,
    )
    for item in merged:
        item["source"] = "hybrid"

    if use_reranking and merged:
        final_results = rerank(
            query,
            merged,
            top_k=top_k,
            method=RERANK_METHOD,
        )
    else:
        final_results = merged[:top_k]

    best_score = final_results[0]["score"] if final_results else 0.0
    if not final_results or best_score < score_threshold:
        fallback = pageindex_search(query, top_k=top_k)
        return fallback if fallback else final_results[:top_k]
    return final_results[:top_k]


if __name__ == "__main__":
    for result in retrieve("hinh phat tang tru ma tuy", top_k=3):
        print(
            f"[{result['score']:.3f}] [{result['source']}] "
            f"{result['content'][:100]}..."
        )
