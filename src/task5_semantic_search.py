"""Task 5 - Semantic search bang model va vector store cua Task 4."""

import json

from .task4_chunking_indexing import (
    COLLECTION_NAME,
    LOCAL_INDEX_PATH,
    connect_weaviate,
    get_embedding_model,
)


def _search_local_index(query: str, top_k: int) -> list[dict]:
    """Cosine search tren JSON index khi Weaviate khong san sang."""
    if not LOCAL_INDEX_PATH.exists():
        return []

    chunks = json.loads(LOCAL_INDEX_PATH.read_text(encoding="utf-8"))
    model = get_embedding_model()
    query_vector = model.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    results = []
    for chunk in chunks:
        if "khong co text layer" in chunk.get("content", "").lower():
            continue
        embedding = chunk.get("embedding", [])
        score = sum(
            query_value * chunk_value
            for query_value, chunk_value in zip(query_vector, embedding)
        )
        results.append(
            {
                "content": chunk.get("content", ""),
                "score": float(score),
                "metadata": chunk.get("metadata", {}),
            }
        )
    return sorted(
        results,
        key=lambda result: result["score"],
        reverse=True,
    )[:top_k]


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Tra ve cac chunk gan query nhat, sap xep theo cosine score giam dan."""
    from weaviate.classes.query import MetadataQuery

    query = query.strip()
    if not query or top_k <= 0:
        return []

    try:
        client = connect_weaviate()
    except Exception:
        return _search_local_index(query, top_k)

    try:
        if not client.collections.exists(COLLECTION_NAME):
            return _search_local_index(query, top_k)

        model = get_embedding_model()
        query_vector = model.encode(
            query,
            normalize_embeddings=True,
        ).tolist()

        collection = client.collections.get(COLLECTION_NAME)
        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=top_k,
            return_metadata=MetadataQuery(distance=True),
        )

        results = []
        for item in response.objects:
            properties = item.properties
            distance = item.metadata.distance
            score = 1.0 - float(distance if distance is not None else 1.0)
            results.append(
                {
                    "content": properties.get("content", ""),
                    "score": score,
                    "metadata": {
                        "source": properties.get("source", ""),
                        "source_path": properties.get("source_path", ""),
                        "type": properties.get("doc_type", ""),
                        "chunk_index": properties.get("chunk_index", 0),
                    },
                }
            )

        return sorted(
            results,
            key=lambda result: result["score"],
            reverse=True,
        )[:top_k]
    finally:
        client.close()


if __name__ == "__main__":
    for result in semantic_search(
        "hinh phat cho toi tang tru ma tuy",
        top_k=5,
    ):
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
