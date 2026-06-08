"""
Task 4 - Chunking, embedding va indexing vao Weaviate.

RecursiveCharacterTextSplitter phu hop voi tap du lieu tron giua van ban phap
luat va bai bao. Chunk 500 ky tu du gon cho mot y, overlap 50 ky tu giup giu
ngu canh tai bien ma khong lam tang qua nhieu so chunk.

all-MiniLM-L6-v2 la model 384 chieu nhe, nhanh va nam trong danh sach goi y
cua de bai. Weaviate duoc chon theo khuyen nghi va chay embedded mac dinh de
khong phu thuoc Docker; van co the chuyen sang local server hoac cloud.
"""

import json
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
VECTOR_DATA_DIR = Path(__file__).parent.parent / "data" / "vector_store"
LOCAL_INDEX_PATH = VECTOR_DATA_DIR / "drug_law_docs.json"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

VECTOR_STORE = "weaviate_with_local_fallback"
COLLECTION_NAME = "DrugLawDocs"


def load_documents() -> list[dict]:
    """Doc toan bo Markdown va tra ve content cung metadata."""
    documents = []
    if not STANDARDIZED_DIR.exists():
        return documents

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content or "khong co text layer" in content.lower():
            continue

        relative_path = md_file.relative_to(STANDARDIZED_DIR)
        doc_type = relative_path.parts[0] if len(relative_path.parts) > 1 else "unknown"
        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "source_path": relative_path.as_posix(),
                    "type": doc_type,
                },
            }
        )
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Tach documents bang RecursiveCharacterTextSplitter."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        length_function=len,
    )

    chunks = []
    for document in documents:
        splits = splitter.split_text(document["content"])
        for index, text in enumerate(splits):
            text = text.strip()
            if not text:
                continue
            chunks.append(
                {
                    "content": text,
                    "metadata": {
                        **document.get("metadata", {}),
                        "chunk_index": index,
                    },
                }
            )
    return chunks


@lru_cache(maxsize=1)
def get_embedding_model():
    """Load model mot lan trong moi process."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Sinh normalized embedding cho toan bo chunks."""
    if not chunks:
        return []

    model = get_embedding_model()
    vectors = model.encode(
        [chunk["content"] for chunk in chunks],
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector.tolist()
    return chunks


def connect_weaviate():
    """Ket noi Weaviate embedded, local server hoac cloud."""
    import weaviate

    load_dotenv()
    url = os.getenv("WEAVIATE_URL", "").strip()
    api_key = os.getenv("WEAVIATE_API_KEY", "").strip()
    mode = os.getenv("WEAVIATE_MODE", "embedded").strip().lower()

    if url:
        if not api_key:
            raise ValueError("WEAVIATE_API_KEY is required with WEAVIATE_URL")
        return weaviate.connect_to_weaviate_cloud(
            cluster_url=url,
            auth_credentials=weaviate.auth.AuthApiKey(api_key),
        )
    if mode == "local":
        return weaviate.connect_to_local()
    if mode != "embedded":
        raise ValueError("WEAVIATE_MODE must be 'embedded' or 'local'")

    VECTOR_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return weaviate.connect_to_embedded(
        persistence_data_path=str(VECTOR_DATA_DIR),
    )


def index_to_vectorstore(chunks: list[dict]) -> int:
    """Tao lai collection va insert chunks kem embedding da cau hinh."""
    from weaviate.classes.config import Configure, DataType, Property

    if not chunks:
        return 0

    try:
        client = connect_weaviate()
    except Exception as error:
        VECTOR_DATA_DIR.mkdir(parents=True, exist_ok=True)
        LOCAL_INDEX_PATH.write_text(
            json.dumps(chunks, ensure_ascii=False),
            encoding="utf-8",
        )
        print(
            f"Weaviate unavailable ({type(error).__name__}); "
            f"saved local index to {LOCAL_INDEX_PATH}"
        )
        return len(chunks)

    try:
        if client.collections.exists(COLLECTION_NAME):
            client.collections.delete(COLLECTION_NAME)

        collection = client.collections.create(
            name=COLLECTION_NAME,
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="content", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="source_path", data_type=DataType.TEXT),
                Property(name="doc_type", data_type=DataType.TEXT),
                Property(name="chunk_index", data_type=DataType.INT),
            ],
        )

        with collection.batch.dynamic() as batch:
            for chunk in chunks:
                metadata = chunk.get("metadata", {})
                batch.add_object(
                    properties={
                        "content": chunk["content"],
                        "source": metadata.get("source", ""),
                        "source_path": metadata.get("source_path", ""),
                        "doc_type": metadata.get("type", ""),
                        "chunk_index": metadata.get("chunk_index", 0),
                    },
                    vector=chunk["embedding"],
                )

        failed = collection.batch.failed_objects
        if failed:
            raise RuntimeError(f"Weaviate failed to index {len(failed)} objects")
        return len(chunks)
    finally:
        client.close()


def run_pipeline() -> int:
    """Chay pipeline load -> chunk -> embed -> index."""
    documents = load_documents()
    if not documents:
        raise RuntimeError(
            "Khong co Markdown trong data/standardized. Hay chay Task 1-3 truoc."
        )

    chunks = chunk_documents(documents)
    chunks = embed_chunks(chunks)
    indexed_count = index_to_vectorstore(chunks)
    print(f"Indexed {indexed_count} chunks to {COLLECTION_NAME}")
    return indexed_count


if __name__ == "__main__":
    run_pipeline()
