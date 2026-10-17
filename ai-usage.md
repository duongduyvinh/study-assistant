# Khai báo dùng AI (`ai-usage.md`)

## 1. Phần do AI hỗ trợ khởi tạo ban đầu
- **Phân tích chiến lược & Khung repo ban đầu (`2683673`):** Khung kiến trúc phân tách module trong `src/t3/`, cấu trúc `tests/`, `prompts/`, `Makefile` và `README.md` được Claude (Anthropic) hỗ trợ sinh bản nháp đầu tiên dựa trên tài liệu đặc tả thiết kế (schema metadata, stateless `RunContext`, luồng Gatekeeper, MCP stdio) do nhóm đề xuất.
- **Dữ liệu và câu hỏi mẫu khởi đầu:** Nội dung mẫu ban đầu của `data/lesson_plans/web201.md` và 14 câu hỏi khởi đầu trong `tests/testset.jsonl` được AI hỗ trợ tạo khung để kiểm thử luồng offline.

## 2. Phần nhóm tự thực hiện, chỉnh sửa và mở rộng
Toàn bộ thành viên đã đọc hiểu, rà soát, chỉnh sửa sâu từng module theo phân công và kiểm thử thực tế:
- **Thái Ngọc Tâm Như (`24110117` / `24110171`) – LLM Client, Agent, Tools, MCP & Logging (Commits: `6201b44`, `665a2c8`, `0de2175`, `c1a2b8b`, `eef9dd2`, `c587de4`):**
  - Mở rộng dataclass `Usage` trong `src/t3/types.py` (bổ sung `calls`, `max_prompt`).
  - Nâng cấp `src/t3/llm.py` và `src/t3/logging_jsonl.py` để ghi nhận `calls`, `max_prompt` và xuất dòng tổng kết `usage_summary` cuối mỗi file log JSONL.
  - Cấu hình và tinh chỉnh `src/t3/judge.py` để gọi Local AI Judge; kết nối và chạy thực nghiệm hệ thống với API thật (`gemini-3.5-flash-lite` và `gemini-embedding-001`), thu thập trace thực tế trong `logs/` và `eval_out/*.csv`.
  - Cập nhật unit tests (`tests/test_llm_http.py`, `tests/test_pipeline.py`) và viết báo cáo kỹ thuật `docs/Nhu.md`.
- **Nguyễn Thị Lan Phương (`24110124`) – Lesson Plan Data, Chunking/Indexing & Evaluation (Commits: `3e78121`, `af241a5`, `a029c72`):**
  - Chuẩn hóa YAML front matter (`title`, `owner: Lan Phương`, `doc_id`, `version`, `semester_start`) và nội dung sư phạm cho `data/lesson_plans/web101.md` và `data/lesson_plans/web201.md`.
  - Mở rộng bộ `tests/testset.jsonl` từ 14 lên 18 câu hỏi (`q15`–`q18`), bao phủ đầy đủ 4 nhãn (`answerable`, `near_miss`, `no_answer`, `ambiguous`) và các tình huống giao thoa chủ đề giữa hai môn.
  - Tách file kiểm thử `tests/test_mcp_eval.py` thành 2 file độc lập `tests/test_eval.py` và `tests/test_mcp.py` để tránh xung đột khi chạy kiểm thử song song; chạy `run_eval.py` xuất kết quả đánh giá và viết báo cáo `docs/phuong.md`.
- **Dương Duy Vinh (`24110148`) – RAG Retrieval, Gatekeeper, Prompts & Pipeline (Commit: `fa6478d`):**
  - Nâng cấp `src/t3/retrieve.py`: tích hợp bước chuẩn hóa điểm số BM25 trước khi dung hợp RRF trong chế độ tìm kiếm hỗn hợp (Hybrid Search).
  - Kiểm chứng toàn bộ luồng `precheck` -> `retrieve` -> `gate` -> `answer_stage` -> `check_citations` cùng 20 bài kiểm thử trong `tests/test_retrieve_gate.py`, `tests/test_pipeline.py` và biên soạn báo cáo kỹ thuật `docs/vinh.md`.

## 3. Phần tái sử dụng từ bài thực hành trên lớp
- **Bài thực hành Tuần 4 (`week4/ex2_1_hybrid_search.py`):**
  - Tái sử dụng hàm chuẩn hóa điểm `normalise_scores(scores: list[float]) -> list[float]` tích hợp vào `src/t3/retrieve.py` (dòng 91–98) và gọi tại hàm `retrieve()` (dòng 111) để chuẩn hóa miền điểm BM25 về đoạn `[0, 1]` trước khi kết hợp với kết quả Dense Vector bằng thuật toán Reciprocal Rank Fusion (RRF, `k=60`).
- **Kiến trúc Agent Loop & Tool Verification (Bài học C2/C3 trên lớp):**
  - Vận dụng mẫu thiết kế vòng lặp ReAct giới hạn bước (`MAX_STEPS = 6`), truyền `RunContext` tường minh không dùng biến toàn cục, bắt lỗi thực thi tool trả về chuỗi `"Error: ..."` cho LLM tự sửa tham số, và áp dụng mẫu kiểm chứng bắt buộc (Write-then-Verify: `save_study_note` -> `list_study_notes`) trong `src/t3/agent.py`.

## 4. Thư viện, Mô hình và Dịch vụ bên ngoài
- **Mô hình Embedding:** Google Gemini `gemini-embedding-001` (phân biệt `RETRIEVAL_DOCUMENT` và `RETRIEVAL_QUERY`) + `FakeEmbedder` tất định dùng cho bộ 56 unit tests offline.
- **Mô hình Ngôn ngữ Lớn (LLM):** `gemini-3.5-flash-lite` (gọi qua endpoint tương thích chuẩn OpenAI trong `src/t3/llm.py` và `src/t3/http.py`).
- **Mô hình Đánh giá Cục bộ (Local LLM-as-a-Judge):** `gemma3:1b` chạy cục bộ qua Ollama (`http://localhost:11434/api/generate`) trong `src/t3/judge.py`.
- **Cơ sở dữ liệu & Thư viện Python:** `qdrant-client` (chế độ `:memory:`), `rank-bm25` (`BM25Okapi`), `mcp` (`1.30.0` - giao thức `stdio`), `httpx`, `pytest`.