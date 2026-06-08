"""
Task 8 - PageIndex vectorless retrieval.

PageIndex cloud nhan PDF, tao tree/OCR va reasoning retrieval. Doc IDs duoc
luu trong manifest de khong upload trung. Khi chua co API key, module dung
structural fallback tren cac section Markdown, khong dung vector embedding.
"""

import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

load_dotenv()

PROJECT_DIR = Path(__file__).parent.parent
LEGAL_DIR = PROJECT_DIR / "data" / "landing" / "legal"
STANDARDIZED_DIR = PROJECT_DIR / "data" / "standardized"
MANIFEST_PATH = PROJECT_DIR / "data" / "pageindex_documents.json"


def _usable_key(value: str) -> bool:
    value = value.strip()
    return bool(value and "xxx" not in value.lower() and not value.endswith("..."))


def _client():
    from pageindex import PageIndexClient

    api_key = os.getenv("PAGEINDEX_API_KEY", "")
    if not _usable_key(api_key):
        raise RuntimeError("PAGEINDEX_API_KEY is missing or still a placeholder")
    return PageIndexClient(api_key=api_key)


def upload_documents() -> dict[str, str]:
    """Upload cac PDF phap luat va luu filename -> doc_id manifest."""
    client = _client()
    manifest = {}
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    for pdf_file in sorted(LEGAL_DIR.glob("*.pdf")):
        if pdf_file.name in manifest:
            continue
        response = client.submit_document(str(pdf_file))
        doc_id = response.get("doc_id")
        if not doc_id:
            raise RuntimeError(f"PageIndex did not return doc_id for {pdf_file.name}")
        manifest[pdf_file.name] = doc_id
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST_PATH.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Uploaded: {pdf_file.name} -> {doc_id}")
    return manifest


def _wait_for_retrieval(client, retrieval_id: str, timeout: int = 120) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = client.get_retrieval(retrieval_id)
        status = str(response.get("status", "")).lower()
        if status in {"completed", "ready", "success"}:
            return response
        if status in {"failed", "error"}:
            raise RuntimeError(f"PageIndex retrieval failed: {response}")
        if any(key in response for key in ("results", "result", "retrieval")):
            return response
        time.sleep(2)
    raise TimeoutError("PageIndex retrieval timed out")


def _extract_result_items(value) -> list[dict]:
    """Tim cac result co text/content trong response co shape thay doi."""
    items = []
    if isinstance(value, dict):
        text = value.get("content") or value.get("text")
        if isinstance(text, str) and text.strip():
            items.append(value)
        for child in value.values():
            items.extend(_extract_result_items(child))
    elif isinstance(value, list):
        for child in value:
            items.extend(_extract_result_items(child))
    return items


def _split_sections() -> list[dict]:
    """Tach Markdown theo heading/paragraph cho structural fallback."""
    sections = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        blocks = re.split(r"\n(?=#{1,6}\s)|\n{2,}", content)
        for index, block in enumerate(blocks):
            block = block.strip()
            if len(block) < 40 or "khong co text layer" in block.lower():
                continue
            sections.append(
                {
                    "content": block,
                    "metadata": {
                        "source": md_file.name,
                        "type": md_file.parent.name,
                        "section_index": index,
                    },
                }
            )
    return sections


def _local_structural_search(query: str, top_k: int) -> list[dict]:
    sections = _split_sections()
    if not sections:
        return []

    tokenize = lambda text: re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    bm25 = BM25Okapi([tokenize(item["content"]) for item in sections])
    scores = bm25.get_scores(tokenize(query))
    indices = sorted(
        range(len(scores)),
        key=lambda index: float(scores[index]),
        reverse=True,
    )
    results = []
    for index in indices:
        score = float(scores[index])
        if score <= 0:
            continue
        results.append(
            {
                **sections[index],
                "score": score,
                "source": "pageindex",
                "pageindex_mode": "local_structural_fallback",
            }
        )
        if len(results) >= top_k:
            break
    return results


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Query moi PageIndex document va gop top results."""
    if not query.strip() or top_k <= 0:
        return []

    try:
        client = _client()
        manifest = (
            json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            if MANIFEST_PATH.exists()
            else upload_documents()
        )
        results = []
        for filename, doc_id in manifest.items():
            if not client.is_retrieval_ready(doc_id):
                continue
            submission = client.submit_query(doc_id, query, thinking=False)
            retrieval_id = submission.get("retrieval_id")
            if not retrieval_id:
                continue
            response = _wait_for_retrieval(client, retrieval_id)
            for rank, item in enumerate(_extract_result_items(response), start=1):
                content = item.get("content") or item.get("text")
                results.append(
                    {
                        "content": content,
                        "score": float(item.get("score", 1.0 / rank)),
                        "metadata": {
                            **(item.get("metadata") or {}),
                            "source": filename,
                            "doc_id": doc_id,
                        },
                        "source": "pageindex",
                        "pageindex_mode": "cloud",
                    }
                )
        return sorted(
            results,
            key=lambda item: item["score"],
            reverse=True,
        )[:top_k] or _local_structural_search(query, top_k)
    except Exception:
        return _local_structural_search(query, top_k)


if __name__ == "__main__":
    for result in pageindex_search("hinh phat su dung ma tuy", top_k=3):
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
