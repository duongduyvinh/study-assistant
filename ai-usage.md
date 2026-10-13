# Khai báo dùng AI

> Mẫu điền sẵn phần mình biết chắc. **Hãy sửa cho khớp thực tế trước khi nộp**: vấn đáp sẽ hỏi từng dòng.

## Phần do AI soạn

- Khung repo và bản nháp đầu của mọi file trong `src/t3/`, `tests/`, `prompts/`, `Makefile`, `README.md` được Claude (Anthropic) soạn từ tài liệu chiến lược và draft thiết kế của nhóm.
- Dữ liệu `data/lesson_plans/web201.md` là nội dung mẫu do AI đặt, không phải lesson plan thật.
- `testset.jsonl` bản đầu (14 câu) do AI soạn từ hai lesson plan mẫu.

## Phần nhóm tự làm hoặc đã sửa

- TODO: ghi rõ file nào nhóm đã đọc, sửa, viết lại; ai chịu trách nhiệm phần nào (A: dữ liệu, RAG, gate, eval; B: LLM client, agent, tool, MCP).
- TODO: lesson plan thật của các môn đang học; testset mở rộng; ngưỡng gate chọn từ eval.

## Phần tái dùng từ bài lớp

- TODO: `w4/ex2_1_hybrid_search.py` có được tái dùng không? Bản nháp hiện tự viết RRF trong `retrieve.py`, chưa dùng file lớp. Nếu thay bằng code lớp thì ghi rõ chỗ nào.
- TODO: bài C2/C3 của agent loop có tham khảo không?

## Thư viện và dịch vụ bên ngoài

- Gemini `gemini-embedding-001` (embedding), một LLM qua API (ghi model thật bạn dùng), Qdrant in-memory, `rank-bm25`, `mcp`, `httpx`, `pytest`.
- Judge cục bộ Gemma 3 1B qua Ollama: chỉ ghi nếu thực sự dùng.
