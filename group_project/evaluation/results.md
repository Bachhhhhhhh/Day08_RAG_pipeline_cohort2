# RAG Evaluation Results

## Framework sử dụng

**Offline deterministic proxy**

DeepEval là framework chính của dự án. Khi chưa có OpenAI judge key hợp lệ, báo
cáo dùng bộ metric proxy minh bạch, chạy hoàn toàn offline để kiểm tra A/B.

## Overall Scores

| Metric | Config A (hybrid + rerank) | Config B (dense-only) | Delta |
|--------|----------------------------|-----------------------|-------|
| Faithfulness | 0.856 | 0.884 | -0.028 |
| Answer Relevance | 0.630 | 0.536 | +0.094 |
| Context Recall | 0.854 | 0.721 | +0.132 |
| Context Precision | 0.810 | 0.750 | +0.060 |
| Average | 0.787 | 0.723 | +0.065 |

## A/B Comparison Analysis

**Config A:** Semantic + BM25 + RRF + Jina reranking.

**Config B:** Semantic search only, no fusion or reranking.

**Kết luận:** Config A tốt hơn trên bộ 20 câu hỏi. Config A đạt
0.787, Config B đạt 0.723. Kết quả
cho biết tác động tổng hợp của BM25, RRF và reranking đối với cả recall lẫn độ
chính xác của context.

## Worst Performers (Bottom 3 của Config A)

| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |
|---|----------|--------------|-----------|--------|---------------|------------|
| 1 | Quy trình cai nghiện ma túy gồm những giai đoạn nào? | 0.618 | 0.257 | 0.262 | Retrieval | Không lấy đủ bằng chứng liên quan đến đáp án chuẩn |
| 2 | Châu Việt Cường bị kết án về tội gì và mức án thay đổi như thế nào sau phiên phúc thẩm? | 0.955 | 0.209 | 0.880 | Generation | Câu trả lời còn thiếu trọng tâm hoặc diễn đạt chưa đủ rõ |
| 3 | Cai nghiện ma túy tự nguyện tại gia đình hoặc cộng đồng kéo dài bao lâu? | 0.684 | 0.682 | 0.667 | Generation | Câu trả lời còn thiếu trọng tâm hoặc diễn đạt chưa đủ rõ |

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
