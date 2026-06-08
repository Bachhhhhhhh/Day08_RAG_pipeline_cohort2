"""Evaluation pipeline for the group RAG project.

DeepEval is the primary judge framework. An offline deterministic evaluator is
included so A/B experiments and reports remain reproducible without a paid LLM
judge key.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from statistics import mean

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from dotenv import load_dotenv

from src.task5_semantic_search import semantic_search
from src.task9_retrieval_pipeline import retrieve

load_dotenv(PROJECT_DIR / ".env")

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"
RAW_RESULTS_PATH = Path(__file__).parent / "evaluation_results.json"

METRIC_NAMES = [
    "faithfulness",
    "answer_relevance",
    "context_recall",
    "context_precision",
]

STOPWORDS = {
    "và", "là", "có", "của", "cho", "được", "những", "các", "một", "theo",
    "trong", "về", "tại", "người", "ma", "túy", "thì", "nào", "bao", "nhiêu",
    "gì", "ai", "không", "đó", "với", "để", "khi", "từ", "đến",
}


def load_golden_dataset() -> list[dict]:
    dataset = json.loads(GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))
    if len(dataset) < 15:
        raise ValueError("Golden dataset must contain at least 15 test cases")

    required = {"question", "expected_answer", "expected_context"}
    for index, item in enumerate(dataset):
        missing = required - item.keys()
        if missing:
            raise ValueError(f"Test case {index} is missing fields: {sorted(missing)}")
    return dataset


def _usable_key(value: str) -> bool:
    value = value.strip()
    return bool(value and "xxx" not in value.lower() and not value.endswith("..."))


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"\w+", text.lower(), flags=re.UNICODE)
        if len(token) > 1 and token not in STOPWORDS
    }


def _overlap(left: str, right: str) -> float:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens)


def _source_label(chunk: dict) -> str:
    return str(chunk.get("metadata", {}).get("source", "")).lower()


def _answer_from_chunks(chunks: list[dict]) -> str:
    if not chunks:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có"

    statements = []
    for chunk in chunks[:3]:
        text = re.sub(r"\s+", " ", chunk.get("content", "")).strip()
        sentences = re.split(r"(?<=[.!?])\s+", text)
        statement = next(
            (sentence for sentence in sentences if 45 <= len(sentence) <= 420),
            text[:420],
        )
        source = chunk.get("metadata", {}).get("source", "Nguồn không rõ")
        statements.append(f"{statement} [{source}]")
    return "\n\n".join(statements)


def run_config(config_name: str, item: dict, top_k: int = 5) -> dict:
    question = item["question"]
    if config_name == "hybrid_rerank":
        chunks = retrieve(
            question,
            top_k=top_k,
            score_threshold=0.0,
            use_reranking=True,
        )
    elif config_name == "dense_only":
        chunks = semantic_search(question, top_k=top_k)
        for chunk in chunks:
            chunk["source"] = "hybrid"
    else:
        raise ValueError(f"Unknown config: {config_name}")

    return {
        "answer": _answer_from_chunks(chunks),
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "none") if chunks else "none",
    }


def score_offline(item: dict, result: dict) -> dict:
    """Transparent proxy metrics for local, key-free evaluation."""
    contexts = [source.get("content", "") for source in result["sources"]]
    combined_context = "\n".join(contexts)
    answer = result["answer"]

    faithfulness = _overlap(answer, combined_context)
    answer_relevance = (
        0.55 * _overlap(item["question"], answer)
        + 0.45 * _overlap(item["expected_answer"], answer)
    )
    context_recall = _overlap(item["expected_answer"], combined_context)

    expected_sources = {
        source.lower() for source in item.get("expected_sources", [])
    }
    useful_chunks = 0
    for source in result["sources"]:
        source_match = any(
            expected in _source_label(source)
            for expected in expected_sources
        )
        content_match = _overlap(item["expected_answer"], source.get("content", ""))
        if source_match or content_match >= 0.18:
            useful_chunks += 1
    context_precision = (
        useful_chunks / len(result["sources"])
        if result["sources"]
        else 0.0
    )

    return {
        "faithfulness": round(min(faithfulness, 1.0), 4),
        "answer_relevance": round(min(answer_relevance, 1.0), 4),
        "context_recall": round(min(context_recall, 1.0), 4),
        "context_precision": round(min(context_precision, 1.0), 4),
    }


def evaluate_config_offline(
    config_name: str,
    golden_dataset: list[dict],
) -> dict:
    cases = []
    for item in golden_dataset:
        result = run_config(config_name, item)
        scores = score_offline(item, result)
        cases.append(
            {
                "id": item.get("id"),
                "question": item["question"],
                "category": item.get("category", "unknown"),
                "difficulty": item.get("difficulty", "unknown"),
                "answer": result["answer"],
                "retrieved_sources": [
                    source.get("metadata", {}).get("source", "")
                    for source in result["sources"]
                ],
                "scores": scores,
                "average": round(mean(scores.values()), 4),
            }
        )

    aggregate = {
        metric: round(mean(case["scores"][metric] for case in cases), 4)
        for metric in METRIC_NAMES
    }
    aggregate["average"] = round(mean(aggregate.values()), 4)
    return {"aggregate": aggregate, "cases": cases}


def evaluate_with_deepeval(
    config_name: str,
    golden_dataset: list[dict],
) -> dict:
    """Evaluate one configuration with the four required DeepEval metrics."""
    if not _usable_key(os.getenv("OPENAI_API_KEY", "")):
        raise RuntimeError("A valid OPENAI_API_KEY is required for DeepEval")

    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
    )
    from deepeval.test_case import LLMTestCase

    cases = []
    for item in golden_dataset:
        result = run_config(config_name, item)
        test_case = LLMTestCase(
            input=item["question"],
            actual_output=result["answer"],
            expected_output=item["expected_answer"],
            retrieval_context=[
                source.get("content", "") for source in result["sources"]
            ],
        )
        metrics = {
            "faithfulness": FaithfulnessMetric(threshold=0.7, async_mode=False),
            "answer_relevance": AnswerRelevancyMetric(threshold=0.7, async_mode=False),
            "context_recall": ContextualRecallMetric(threshold=0.7, async_mode=False),
            "context_precision": ContextualPrecisionMetric(
                threshold=0.7,
                async_mode=False,
            ),
        }
        scores = {}
        reasons = {}
        for name, metric in metrics.items():
            metric.measure(test_case)
            scores[name] = round(float(metric.score or 0.0), 4)
            reasons[name] = metric.reason or ""

        cases.append(
            {
                "id": item.get("id"),
                "question": item["question"],
                "category": item.get("category", "unknown"),
                "difficulty": item.get("difficulty", "unknown"),
                "answer": result["answer"],
                "retrieved_sources": [
                    source.get("metadata", {}).get("source", "")
                    for source in result["sources"]
                ],
                "scores": scores,
                "reasons": reasons,
                "average": round(mean(scores.values()), 4),
            }
        )

    aggregate = {
        metric: round(mean(case["scores"][metric] for case in cases), 4)
        for metric in METRIC_NAMES
    }
    aggregate["average"] = round(mean(aggregate.values()), 4)
    return {"aggregate": aggregate, "cases": cases}


def compare_configs(
    golden_dataset: list[dict],
    mode: str = "auto",
) -> dict:
    use_deepeval = mode == "deepeval" or (
        mode == "auto" and _usable_key(os.getenv("OPENAI_API_KEY", ""))
    )
    evaluator = evaluate_with_deepeval if use_deepeval else evaluate_config_offline
    return {
        "framework": "DeepEval" if use_deepeval else "Offline deterministic proxy",
        "config_a": {
            "name": "hybrid_rerank",
            "description": "Semantic + BM25 + RRF + Jina reranking",
            **evaluator("hybrid_rerank", golden_dataset),
        },
        "config_b": {
            "name": "dense_only",
            "description": "Semantic search only, no fusion or reranking",
            **evaluator("dense_only", golden_dataset),
        },
    }


def _failure_stage(case: dict) -> tuple[str, str]:
    scores = case["scores"]
    if scores["context_recall"] < 0.35:
        return "Retrieval", "Không lấy đủ bằng chứng liên quan đến đáp án chuẩn"
    if scores["context_precision"] < 0.4:
        return "Retrieval", "Top-k chứa nhiều đoạn không trực tiếp hữu ích"
    if scores["faithfulness"] < 0.6:
        return "Generation", "Câu trả lời chưa bám sát nội dung context"
    return "Generation", "Câu trả lời còn thiếu trọng tâm hoặc diễn đạt chưa đủ rõ"


def export_results(comparison: dict) -> None:
    config_a = comparison["config_a"]
    config_b = comparison["config_b"]
    aggregate_a = config_a["aggregate"]
    aggregate_b = config_b["aggregate"]
    worst = sorted(config_a["cases"], key=lambda case: case["average"])[:3]

    labels = {
        "faithfulness": "Faithfulness",
        "answer_relevance": "Answer Relevance",
        "context_recall": "Context Recall",
        "context_precision": "Context Precision",
        "average": "Average",
    }
    rows = []
    for metric in [*METRIC_NAMES, "average"]:
        delta = aggregate_a[metric] - aggregate_b[metric]
        rows.append(
            f"| {labels[metric]} | {aggregate_a[metric]:.3f} | "
            f"{aggregate_b[metric]:.3f} | {delta:+.3f} |"
        )

    worst_rows = []
    for index, case in enumerate(worst, start=1):
        stage, cause = _failure_stage(case)
        scores = case["scores"]
        question = case["question"].replace("|", "/")
        worst_rows.append(
            f"| {index} | {question} | {scores['faithfulness']:.3f} | "
            f"{scores['answer_relevance']:.3f} | "
            f"{scores['context_recall']:.3f} | {stage} | {cause} |"
        )

    winner = (
        "Config A tốt hơn"
        if aggregate_a["average"] >= aggregate_b["average"]
        else "Config B tốt hơn"
    )
    content = f"""# RAG Evaluation Results

## Framework sử dụng

**{comparison['framework']}**

DeepEval là framework chính của dự án. Khi chưa có OpenAI judge key hợp lệ, báo
cáo dùng bộ metric proxy minh bạch, chạy hoàn toàn offline để kiểm tra A/B.

## Overall Scores

| Metric | Config A (hybrid + rerank) | Config B (dense-only) | Delta |
|--------|----------------------------|-----------------------|-------|
{chr(10).join(rows)}

## A/B Comparison Analysis

**Config A:** {config_a['description']}.

**Config B:** {config_b['description']}.

**Kết luận:** {winner} trên bộ {len(config_a['cases'])} câu hỏi. Config A đạt
{aggregate_a['average']:.3f}, Config B đạt {aggregate_b['average']:.3f}. Kết quả
cho biết tác động tổng hợp của BM25, RRF và reranking đối với cả recall lẫn độ
chính xác của context.

## Worst Performers (Bottom 3 của Config A)

| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |
|---|----------|--------------|-----------|--------|---------------|------------|
{chr(10).join(worst_rows)}

## Recommendations

### Cải tiến 1
**Action:** Bổ sung bản text hoặc OCR chất lượng cao cho các nghị định đang là PDF scan.

**Expected impact:** Tăng context recall cho câu hỏi về danh mục chất và hướng dẫn thi hành.

### Cải tiến 2
**Action:** Tách văn bản pháp luật theo Điều, khoản và lưu số điều trong metadata.

**Expected impact:** Giảm chunk nhiễu, tăng context precision và citation chính xác.

### Cải tiến 3
**Action:** Chạy Jina Reranker và DeepEval bằng API key thật trước buổi demo.

**Expected impact:** Có điểm cross-encoder và LLM judge chính thức thay cho fallback offline.
"""
    RESULTS_PATH.write_text(content, encoding="utf-8")
    RAW_RESULTS_PATH.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["auto", "deepeval", "offline"],
        default="auto",
        help="auto uses DeepEval when a valid OpenAI key exists",
    )
    args = parser.parse_args()

    dataset = load_golden_dataset()
    comparison = compare_configs(dataset, mode=args.mode)
    export_results(comparison)
    print(
        f"Evaluated {len(dataset)} cases with {comparison['framework']}. "
        f"Report: {RESULTS_PATH}"
    )


if __name__ == "__main__":
    main()
