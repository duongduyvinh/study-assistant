# Báo cáo Nhiệm vụ & Chiến lược Cắt Chunk (Lan Phương)

- **Người thực hiện:** Lan Phương
- **Phần việc phụ trách:** Module Agent & LLM (12 điểm) + Chuẩn hóa Dữ liệu Lesson Plan, Chiến lược Chunking & Mở rộng Testset / Tách Test

---

## 1. Phạm vi hệ thống & Người dùng mục tiêu (Scope & Target Users)

### 1.1. Phạm vi hệ thống
Hệ thống `t3-lesson-assistant` là trợ lý học tập thông minh dựa trên giáo trình / kế hoạch bài giảng (lesson plan) dạng Markdown của các môn học (`WEB101` – Lập trình web cơ bản, `WEB201` – Lập trình web nâng cao). Hệ thống hỗ trợ:
- Trả lời chính xác nội dung học theo tuần, khái niệm lý thuyết, bài tập thực hành và tài liệu liên quan.
- Trích dẫn nguồn minh bạch qua mã định danh chunk `[chunk_id]` (ví dụ `[WEB101-w01-concept]`), tự động phát hiện và cảnh báo trích dẫn ảo (hallucination).
- Cổng phân loại rủi ro (`gate.py`): Từ chối các câu hỏi ngoài chương trình (`abstain`) hoặc yêu cầu bổ sung thông tin khi thiếu mã môn học (`clarify`).
- Trợ lý Agent hỗ trợ sinh viên lưu và tra cứu ghi chú học tập cá nhân (`save_study_note`, `list_study_notes`), bắt buộc bước kiểm chứng (verify) sau khi ghi dữ liệu.

### 1.2. Người dùng mục tiêu
- **Sinh viên:** Cần tra cứu nhanh lộ trình học tập từng tuần, yêu cầu bài tập cần nộp, ôn tập khái niệm cốt lõi, lưu ghi chú ôn thi.
- **Giảng viên & Trợ giảng:** Cần kiểm tra tiến độ giảng dạy, outline môn học, kiểm tra tính nhất quán giữa bài giảng và bài tập.

---

## 2. Kiến trúc tổng thể & Luồng xử lý (Architecture & Pipeline)

1. **Khởi tạo & Tiền xử lý (Parse & Start):** Chuẩn hóa mã môn học, trích xuất số tuần (`tuần 2`, hoặc `tuần này` dựa trên ngày khai giảng `semester_start`), nhận diện loại mục hỏi (`bài tập`, `khái niệm`...).
2. **Tiền kiểm tra (Precheck):** Nếu câu hỏi thiếu mã môn học hoặc mã môn không hợp lệ, lập tức trả về hành động `clarify` mà không tốn chi phí truy xuất vector hay gọi LLM.
3. **Truy xuất thông tin (Retrieve):**
   - *Week-scan:* Nếu có thông tin tuần cụ thể, trích xuất toàn bộ chunk của tuần đó theo thứ tự sư phạm (`concept` -> `activity` -> `assessment` -> `reference`) với `recall = 1.0`.
   - *Hybrid / Dense:* Nếu không có tuần cụ thể, lọc cứng theo metadata `course_id`, sau đó kết hợp tìm kiếm ngữ nghĩa Dense (Qdrant in-memory vector) và từ khóa BM25 (RRF fusion).
4. **Cổng kiểm soát (Gate Decision):** Tính điểm `rerank_score = 0.7 * cosine + 0.3 * keyword_coverage`. Nếu dưới ngưỡng `GATE_THRESHOLD`, hệ thống từ chối (`abstain`) với lý do rõ ràng.
5. **Sinh câu trả lời & Kiểm chứng (LLM & Citation Check):** Chỉ nhánh `answer` mới gọi LLM với prompt chặt chẽ, bắt buộc trích dẫn bằng mã `[chunk_id]` có thực trong bằng chứng (evidence).

---

## 3. Lý do và Chiến lược cắt Chunk (Chunking Strategy Rationale)

Đây là quyết định thiết kế cốt lõi trong khâu tiền xử lý (`src/t3/ingest.py`):

### 3.1. Cấu trúc cắt chunk theo ngữ nghĩa sư phạm
Thay vì cắt theo số ký tự hoặc token cố định (Fixed-size Chunking, ví dụ 300 token), hệ thống chia nhỏ văn bản dựa trên cấu trúc tự nhiên của giáo trình:
- **Cấp 1 (Phạm vi thời gian & Chủ đề):** Dựa trên tiêu đề tuần `## Tuần <N>: <Chủ đề>` để phân định tuần học.
- **Cấp 2 (Loại nội dung sư phạm):** Dựa trên tiêu đề mục `### Khái niệm` (`concept`), `### Hoạt động` (`activity`), `### Bài tập` (`assessment`), `### Tài liệu` (`reference`).

### 3.2. Vì sao KHÔNG dùng phương pháp cắt cố định 300 token?
1. **Bảo toàn tính toàn vẹn ngữ nghĩa (Semantic Integrity):**
   - Cắt fixed-size rất dễ làm đứt đoạn một danh sách cú pháp (ví dụ cắt đôi thẻ HTML `<ul>`, `<ol>`), làm đứt đoạn một đề bài tập lập trình hay ngắt ngang một khối code/thuật toán giữa chừng. Điều này khiến vector nhúng của đoạn bị cắt mất ngữ cảnh và LLM không đọc trọn vẹn được đề bài.
   - Cắt theo mục sư phạm đảm bảo mỗi chunk là một đơn vị tri thức hoàn chỉnh, khép kín về mặt ý nghĩa.
2. **Khai thác tối đa Lọc Metadata (Hard Metadata Filtering):**
   - Mỗi chunk có metadata xác định: `course_id`, `week`, `topic`, `type`, `clo_ids`, `doc_id`, `version`, `owner`.
   - Khi người dùng hỏi *"Bài tập tuần 2 môn WEB101 là gì?"*, hệ thống chỉ cần áp bộ lọc `course_id == "WEB101" AND week == 2 AND type == "assessment"`, lấy trúng đích 100% tài liệu mà không bị nhiễu bởi các đoạn lý thuyết hay bài tập của tuần khác.
3. **Mã định danh trích dẫn bất biến (`chunk_id`):**
   - Quy tắc đặt ID: `{course_id}-w{week:02d}-{section}` (ví dụ: `WEB101-w01-concept`, `WEB101-w01-assessment`).
   - Khác với số thứ tự tương đối `[1]`, `[2]` (dễ bị đảo lộn khi Agent thực hiện nhiều bước hoặc khi gọi nhiều tool), mã `[chunk_id]` là cố định và mang đầy đủ thông tin xuất xứ. Hệ thống dễ dàng đối chiếu `set(prompt_chunk_ids) >= set(answer_chunk_ids)` để loại bỏ hoàn toàn hiện tượng bịa nguồn trích dẫn.

---

## 4. Cập nhật Dữ liệu Bài giảng & Mở rộng Testset (18 câu)

### 4.1. Cập nhật Lesson Plan (`data/lesson_plans/`)
- Cả hai file `web101.md` và `web201.md` đã được bổ sung trường `title` vào front matter và cập nhật `owner: Lan Phương` (thay thế placeholder `<giảng viên>` bằng người phụ trách thực tế).
- Giữ nguyên `doc_id` (`web101-lesson-plan`, `web201-lesson-plan`) và `version: 2026.1` để đảm bảo tương thích ngược với mọi bài kiểm thử.

### 4.2. Mở rộng bộ Testset (`tests/testset.jsonl`) lên 18 câu
Bộ testset được nâng cấp từ 14 lên 18 câu hỏi:
- **Bảo toàn các dòng then chốt:** Giữ nguyên `q01` (khóa week_scan), `q07` (khóa abstain ngoài tuần), `q13` (khóa clarify thiếu môn) và nhóm 4 câu `no_answer` ngoài chương trình (`q09`–`q12`).
- **Bổ sung các câu từ `q15` đến `q18`:**
  - `q15` (`answerable`): Hỏi bài tập tuần 1 môn `WEB201` (`to-do list`, `DOM`).
  - `q16` (`near_miss`): Hỏi TypeScript interface trong môn `WEB101` (liên quan công nghệ web nhưng ngoài giáo trình -> `abstain`).
  - `q17` (`no_answer`): Hỏi luật chơi cờ vua (ngoài chương trình đào tạo -> `abstain`).
  - `q18` (`ambiguous`): Hỏi nội dung ôn tập tuần 1 nhưng thiếu mã môn học (`course_id: null` -> `clarify`).
- Đảm bảo 100% câu hỏi thuộc đúng 4 nhãn chuẩn: `answerable`, `near_miss`, `no_answer`, `ambiguous` để `test_eval.py` kiểm định chính xác.

---

## 5. Hướng dẫn Chạy Kiểm thử (Verification)

Mọi thay đổi đã được chạy thử nghiệm thực tế với môi trường ảo `.venv`:
```bash
# Chạy toàn bộ 56 test offline (Embedder giả & Mock LLM):
.\.venv\Scripts\python.exe -m pytest -q

# Chạy riêng bộ test đánh giá và MCP:
.\.venv\Scripts\python.exe -m pytest tests/test_eval.py tests/test_mcp.py

# Chạy module đánh giá trên testset:
.\.venv\Scripts\python.exe -m t3.run_eval
```
Kết quả kiểm tra đạt **56/56 tests passed** (100%), xác nhận hệ thống hoạt động ổn định và sẵn sàng cho các mốc phối hợp tiếp theo.
