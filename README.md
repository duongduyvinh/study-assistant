# T3 – Trợ lý học tập trên lesson plan

Hỏi đáp về lesson plan của các môn, có trích dẫn nguồn và biết từ chối câu ngoài chương trình. Mỗi câu hỏi là một luồng độc lập, không có memory và không hỏi lại tiếp.

> Đây là **bản nháp khung**. Phần chạy được offline đã có test; phần gọi API thật (Gemini, LLM) **chưa chạy với key thật**. Xem mục "Chưa kiểm chứng" bên dưới trước khi tin bất kỳ con số nào.

## Chạy

```bash
cp .env.example .env     # điền GEMINI_API_KEY, LLM_API_KEY, kiểm tra LLM_BASE_URL và LLM_MODEL
make setup && make demo  # tạo venv + index, rồi chạy kịch bản mẫu (ghi logs/*.jsonl)
make eval                # dense so với hybrid, quét ngưỡng gate, ghi eval_out/*.csv
make test                # 56 test offline, không cần key
```

Chạy lẻ: `python -m t3.cli ask "Tuần 2 học gì?" --course WEB101`, `python -m t3.cli agent "..." --course WEB101 [--backend mcp]`, `python -m t3.mcp_client`, `python -m t3.run_eval --with-llm`.

## Bản đồ file (mỗi bước pipeline một file)

> **Phân công thành viên**: **A** = Phương, **B** = Bạn, **C** = Vinh.

| File | Việc | Mục đề | Phụ trách |
| --- | --- | --- | --- |
| `data/lesson_plans/*.md` | Lesson plan có front matter (doc_id, course_id, version, owner) | M1 | Phương |
| `src/t3/types.py` | Dataclass: Chunk, Query, Filters, SearchHit, GateDecision, Usage, ToolCall, AgentStep, RunContext... | M2 | chung |
| `src/t3/config.py` | Đọc `.env` và biến môi trường | M0 | Như |
| `src/t3/http.py`, `llm.py` | `call_llm`: timeout, retry backoff, log HTTP, trả Usage; prompt ở `prompts/*.txt` | M3 | Như |
| `src/t3/embeddings.py` | Gemini embedder (taskType tài liệu/câu hỏi), embedder giả để test | M4 | Phương |
| `src/t3/ingest.py`, `text.py` | Chunk theo (tuần, loại), tokenize tiếng Việt cho BM25 | M4 | Phương |
| `src/t3/index.py` | `chunks.jsonl` (chunk + vector), nạp vào Qdrant `:memory:` + BM25 mỗi process | M4 | Phương |
| `src/t3/retrieve.py` | Lọc metadata trước; week-scan, dense, hybrid (RRF) | M4 | Vinh |
| `src/t3/gate.py` | Gate: `precheck` (clarify) và `decide` (rerank + ngưỡng, hàm thuần) | M4 | Vinh |
| `src/t3/prompts.py` | Nạp prompt, dựng evidence, kiểm trích dẫn `[chunk_id]` | M3/M4 | Vinh |
| `src/t3/pipeline.py` | Một request: parse → precheck → retrieve → gate → LLM → kiểm trích dẫn | M4 | Vinh |
| `src/t3/tools.py`, `notes.py` | 5 tool dùng chung, trong đó `save_study_note` (ghi) + `list_study_notes` (verify) | M6 | Như |
| `src/t3/agent.py` | Agent loop tự viết: MAX_STEPS, parse action an toàn, tool lỗi trả `Error: ...`, ép verify sau khi ghi | M5 | Như |
| `src/t3/mcp_server.py`, `mcp_client.py` | MCP stdio, stdout sạch; client discover bằng `list_tools()`, cũng là backend cho agent | M7 | Như |
| `src/t3/run_eval.py`, `judge.py` | recall@k, abstain P/R, từ chối nhầm, citation, token; quét ngưỡng; judge Gemma tuỳ chọn | M8 | Phương (A) |
| `src/t3/logging_jsonl.py`, `cli.py` | Log mỗi run một file jsonl; `setup`/`ask`/`agent`/`demo` | M0 | Như |
| `tests/` | 56 test + `testset.jsonl` (14 câu khởi đầu) | M8 | chung |

## Một request đi thế nào

1. `start`: chuẩn hoá mã môn, tìm số tuần (`tuần 2`, hoặc `tuần này` tính từ `semester_start`), đoán loại hỏi (bài tập, tài liệu...).
2. `precheck`: thiếu hoặc sai mã môn thì `clarify`, không truy xuất.
3. `retrieve`: có tuần thì lấy hết chunk của tuần (không xếp hạng); không thì dense hoặc hybrid với filter môn.
4. `gate.decide`: `rerank_score = 0.7 * cosine + 0.3 * độ phủ từ khoá`, so với `GATE_THRESHOLD`. Dưới ngưỡng thì `abstain`.
5. Chỉ nhánh `answer` gọi LLM; sau đó mọi `[chunk_id]` trong câu trả lời phải nằm trong evidence, sai thì nhắc một lần.
6. `RunContext` ghi log jsonl rồi bỏ. Nhánh không gọi LLM vẫn có dòng `usage` bằng 0.

## Khác với tài liệu thiết kế (artifact)

- Trích dẫn dùng `[chunk_id]` (ví dụ `[WEB101-w02-concept]`) thay vì `[1]`, `[2]`: số thứ tự restart mỗi lần gọi tool nên không dùng được cho agent nhiều bước. Vẫn kiểm bằng tập id đã thấy nên LLM không bịa được nguồn.
- `GateDecision.clarify_hint` đổi thành `hint` vì abstain cũng có gợi ý.
- `mcp` ghim `1.30.0`: bản 2.x đổi `FastMCP` thành `MCPServer`. Nếu lớp bạn dùng 2.x, sửa `mcp_server.py` cho khớp.

## Chưa kiểm chứng (đọc trước khi nộp)

- **Gemini embedder** (`embeddings.py`) và **`call_llm`** mới test bằng server giả (`httpx.MockTransport`), chưa gọi API thật. Kiểm tra tên field theo tài liệu hiện hành của từng provider.
- **ID model** `qwen/qwen3.8-27b` và `LLM_BASE_URL` trong `.env.example` là giả định; dòng Qwen3 có thể chèn khối `<think>` (code đã cắt) hoặc cần tham số tắt suy nghĩ qua `LLM_EXTRA_BODY`.
- **Ngưỡng gate** mặc định 0.5 là chỗ giữ chỗ. Chọn từ bảng quét của `make eval` trên embedding thật.
- **Testset 14 câu** chỉ để chạy được pipeline. Mở rộng lên 18 đến 20 câu, nhiều near-miss và câu hai môn chồng chủ đề. Các số liệu từ `EMBEDDER=fake` không dùng cho báo cáo.
- **Dữ liệu**: `web101.md` lấy từ draft của bạn (`owner`, `semester_start` là giả định); `web201.md` là mẫu tự đặt. Viết lesson plan thật từ môn các bạn đang học.
- **Judge Gemma** (`judge.py`) chưa chạy vì cần Ollama. `make eval` không phụ thuộc vào nó.
- **Dense so với hybrid** chỉ khác nhau rõ ở câu không nêu tuần; câu nêu tuần được lọc cứng nên recall luôn 1. Báo cáo cột `recall@k_no_week_scan`.
- Chưa làm: so chunk theo tuần với cắt cố định 300 token, đồ thị từ CSV (điểm cộng +1 mỗi mục).
- File `data/notes.json` ghi không khoá liên process (chỉ ghi nguyên tử); đủ cho demo, không dành cho nhiều người ghi cùng lúc.
