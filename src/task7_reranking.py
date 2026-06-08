"""
Task 7 - Reranking bang Jina Reranker API.

Model duoc chon theo de bai: jina-reranker-v2-base-multilingual, phu hop voi
du lieu tieng Viet. Neu API key chua duoc cau hinh hoac API tam loi, module
dung local relevance fallback de pipeline khong bi dung.
"""

import os
import re

import requests
from dotenv import load_dotenv

load_dotenv()

JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"
JINA_MODEL = "jina-reranker-v2-base-multilingual"


def _usable_key(value: str) -> bool:
    value = value.strip()
    return bool(value and "xxx" not in value.lower() and not value.endswith("..."))


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


def _local_fallback(
    query: str,
    candidates: list[dict],
    top_k: int,
) -> list[dict]:
    """Fallback ket hop keyword overlap va retrieval rank ban dau."""
    query_tokens = _tokens(query)
    rescored = []
    total = max(len(candidates), 1)

    for rank, candidate in enumerate(candidates):
        doc_tokens = _tokens(candidate.get("content", ""))
        overlap = (
            len(query_tokens & doc_tokens) / len(query_tokens)
            if query_tokens
            else 0.0
        )
        rank_prior = 1.0 - rank / total
        score = 0.75 * overlap + 0.25 * rank_prior
        item = candidate.copy()
        item["score"] = float(score)
        item["reranker"] = "local_fallback"
        rescored.append(item)

    return sorted(
        rescored,
        key=lambda item: item["score"],
        reverse=True,
    )[:top_k]


def rerank_cross_encoder(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """Rerank candidates bang Jina multilingual cross-encoder."""
    if not candidates or top_k <= 0:
        return []

    api_key = os.getenv("JINA_API_KEY", "")
    if not _usable_key(api_key):
        return _local_fallback(query, candidates, top_k)

    try:
        response = requests.post(
            JINA_RERANK_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": JINA_MODEL,
                "query": query,
                "documents": [item.get("content", "") for item in candidates],
                "top_n": min(top_k, len(candidates)),
                "return_documents": False,
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        reranked = []
        for result in payload.get("results", []):
            index = int(result["index"])
            if not 0 <= index < len(candidates):
                continue
            item = candidates[index].copy()
            item["score"] = float(result["relevance_score"])
            item["reranker"] = JINA_MODEL
            reranked.append(item)
        return reranked[:top_k] or _local_fallback(query, candidates, top_k)
    except (requests.RequestException, KeyError, TypeError, ValueError):
        return _local_fallback(query, candidates, top_k)


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """Chon candidates can bang relevance va diversity bang MMR."""
    if not 0 <= lambda_param <= 1:
        raise ValueError("lambda_param must be between 0 and 1")

    def cosine(left: list[float], right: list[float]) -> float:
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = sum(value * value for value in left) ** 0.5
        right_norm = sum(value * value for value in right) ** 0.5
        return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0

    remaining = [
        index
        for index, item in enumerate(candidates)
        if item.get("embedding")
    ]
    selected = []
    results = []

    while remaining and len(results) < top_k:
        best_index = None
        best_score = float("-inf")
        for index in remaining:
            embedding = candidates[index]["embedding"]
            relevance = cosine(query_embedding, embedding)
            redundancy = max(
                (
                    cosine(embedding, candidates[selected_index]["embedding"])
                    for selected_index in selected
                ),
                default=0.0,
            )
            score = lambda_param * relevance - (1 - lambda_param) * redundancy
            if score > best_score:
                best_index = index
                best_score = score

        selected.append(best_index)
        remaining.remove(best_index)
        item = candidates[best_index].copy()
        item["score"] = float(best_score)
        results.append(item)
    return results


def rerank_rrf(
    ranked_lists: list[list[dict]],
    top_k: int = 5,
    k: int = 60,
) -> list[dict]:
    """Gop nhieu ranked list bang Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            key = item.get("content", "")
            if not key:
                continue
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            items.setdefault(key, item)

    results = []
    for content, score in sorted(
        scores.items(),
        key=lambda pair: pair[1],
        reverse=True,
    )[:top_k]:
        item = items[content].copy()
        item["score"] = float(score)
        item["fusion_score"] = float(score)
        results.append(item)
    return results


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",
) -> list[dict]:
    """Unified reranking interface; cross_encoder su dung Jina."""
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    if method == "rrf":
        return rerank_rrf([candidates], top_k)
    if method == "mmr":
        raise ValueError("MMR requires query_embedding; call rerank_mmr directly")
    raise ValueError(f"Unknown rerank method: {method}")
