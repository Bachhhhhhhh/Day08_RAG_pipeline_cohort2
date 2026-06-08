"""Task 10 - RAG generation co citation va document reordering."""

import os
import re

from dotenv import load_dotenv

from .task9_retrieval_pipeline import retrieve

load_dotenv()

# 5 chunks thuong du evidence nhung van gon de han che lost-in-the-middle.
TOP_K = 5
# top_p=0.9 giu cach dien dat tu nhien; temperature=0.3 uu tien tinh factual.
TOP_P = 0.9
TEMPERATURE = 0.3
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

SYSTEM_PROMPT = """Answer the question comprehensively in Vietnamese.
Use only the supplied context. Every factual claim must immediately include a
citation in the form [Source, Year]. If the context does not contain enough
evidence, say exactly: "Tôi không thể xác minh thông tin này từ nguồn hiện có".
Do not invent legal provisions, people, dates, or citations."""


def _usable_key(value: str) -> bool:
    value = value.strip()
    return bool(value and "xxx" not in value.lower() and not value.endswith("..."))


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Dat chunk quan trong nhat o dau, thu hai o cuoi prompt."""
    if len(chunks) <= 2:
        return list(chunks)

    ordered = []
    left = 0
    right = len(chunks) - 1
    while left <= right:
        ordered.append(chunks[left])
        left += 2
    right = len(chunks) - 1 if len(chunks) % 2 == 0 else len(chunks) - 2
    while right >= 1:
        ordered.append(chunks[right])
        right -= 2
    return ordered


def _source_and_year(chunk: dict, index: int) -> tuple[str, str]:
    metadata = chunk.get("metadata", {})
    source = (
        metadata.get("source")
        or metadata.get("title")
        or f"Source {index}"
    )
    year_match = re.search(r"\b(19|20)\d{2}\b", str(source))
    if not year_match:
        year_match = re.search(
            r"\b(19|20)\d{2}\b",
            chunk.get("content", "")[:500],
        )
    year = year_match.group(0) if year_match else "khong ro nam"
    return str(source), year


def format_context(chunks: list[dict]) -> str:
    """Format context kem source/year de LLM tao citation chinh xac."""
    parts = []
    for index, chunk in enumerate(chunks, start=1):
        source, year = _source_and_year(chunk, index)
        doc_type = chunk.get("metadata", {}).get("type", "unknown")
        parts.append(
            f"[Document {index} | Source: {source} | Year: {year} | "
            f"Type: {doc_type}]\n{chunk.get('content', '')}"
        )
    return "\n\n---\n\n".join(parts)


def _extractive_answer(chunks: list[dict]) -> str:
    """Tao answer co citation khi OpenAI key chua san sang."""
    if not chunks:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có"

    claims = []
    for index, chunk in enumerate(chunks[:3], start=1):
        source, year = _source_and_year(chunk, index)
        text = re.sub(r"\s+", " ", chunk.get("content", "")).strip()
        sentences = re.split(r"(?<=[.!?])\s+", text)
        claim = next((sentence for sentence in sentences if len(sentence) >= 40), "")
        if claim:
            claims.append(f"{claim} [{source}, {year}]")
    return (
        "\n\n".join(claims)
        if claims
        else "Tôi không thể xác minh thông tin này từ nguồn hiện có"
    )


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """Retrieve, reorder, format va generate answer co citation."""
    chunks = retrieve(query, top_k=top_k)
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    api_key = os.getenv("OPENAI_API_KEY", "")

    if chunks and _usable_key(api_key):
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=OPENAI_MODEL,
            instructions=SYSTEM_PROMPT,
            input=f"Context:\n{context}\n\nQuestion: {query}",
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        answer = response.output_text.strip()
    else:
        answer = _extractive_answer(reordered)

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": (
            chunks[0].get("source", "hybrid") if chunks else "none"
        ),
        "model": OPENAI_MODEL if _usable_key(api_key) else "extractive_fallback",
    }


if __name__ == "__main__":
    result = generate_with_citation(
        "Hinh phat cho toi tang tru trai phep chat ma tuy?"
    )
    print(result["answer"])
