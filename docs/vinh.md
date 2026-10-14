# Báo cáo Kỹ thuật: Thiết kế & Hiện thực Tầng RAG, Gatekeeper và Pipeline

**Người thực hiện:** Duy Vinh  
**Phạm vi phụ trách:** Tầng RAG (Truy xuất dữ liệu kết hợp), Bộ kiểm soát Gatekeeper, Prompt & Luồng điều phối Pipeline  

---

## 1. Tổng quan & Phạm vi phụ trách

Trong hệ thống trợ lý học tập hỗ trợ giải đáp đề cương bài giảng môn học, tôi chịu trách nhiệm thiết kế và hiện thực toàn bộ tầng **Truy xuất thông tin (RAG)** và **Kiểm soát chất lượng câu trả lời (Gatekeeper & Pipeline)**. Hệ thống được xây dựng với mục tiêu:
1. **Truy xuất chính xác (High Precision & Recall):** Kết hợp linh hoạt giữa quét trọn vẹn tuần học (Week-Scan) và tìm kiếm hỗn hợp (Hybrid Search: Dense Vector + Sparse BM25).
2. **Chống ảo giác tuyệt đối (Zero Hallucination Guardrails):** Kiểm soát câu hỏi qua hai nấc Gatekeeper để từ chối (`abstain`) hoặc yêu cầu làm rõ (`clarify`) khi câu hỏi nằm ngoài phạm vi đề cương, không để mô hình suy diễn kiến thức ngoài giáo trình.
3. **Minh bạch nguồn thông tin (Strict Citation):** Cưỡng chế mọi luận điểm trả lời phải đính kèm mã trích dẫn nguồn theo định dạng `[chunk_id]` (ví dụ: `[WEB101-w02-concept]`).

### Danh sách các module xây dựng:
* [`src/t3/retrieve.py`](file:///d:/aipr1/study-assistant/src/t3/retrieve.py): Xử lý truy xuất 2 pha (Week-scan và Hybrid RRF kết hợp chuẩn hoá điểm).
* [`src/t3/gate.py`](file:///d:/aipr1/study-assistant/src/t3/gate.py): Bộ phán quyết Gatekeeper gồm `precheck` trước truy xuất và `decide` xếp hạng lại theo ngưỡng.
* [`src/t3/prompts.py`](file:///d:/aipr1/study-assistant/src/t3/prompts.py): Trình nạp prompt, render evidence và kiểm tra cú pháp trích dẫn.
* [`prompts/system.txt`](file:///d:/aipr1/study-assistant/prompts/system.txt) & [`prompts/answer.txt`](file:///d:/aipr1/study-assistant/prompts/answer.txt): Bộ chỉ dẫn prompt cho mô hình ngôn ngữ.
* [`src/t3/pipeline.py`](file:///d:/aipr1/study-assistant/src/t3/pipeline.py): Động cơ điều phối toàn bộ vòng đời của một truy vấn (Start $\rightarrow$ Precheck $\rightarrow$ Retrieve $\rightarrow$ Gate $\rightarrow$ LLM $\rightarrow$ Citation Check $\rightarrow$ Output).
* [`tests/test_retrieve_gate.py`](file:///d:/aipr1/study-assistant/tests/test_retrieve_gate.py) & [`tests/test_pipeline.py`](file:///d:/aipr1/study-assistant/tests/test_pipeline.py): Bộ 20 bài kiểm thử chuyên sâu cho toàn bộ tầng RAG và Pipeline.

---

## 2. Kiến trúc Kỹ thuật & Chi tiết Thiết kế

```
                 [ Câu hỏi người dùng ]
                            │
                            ▼
              ┌───────────────────────────┐
              │      pipeline.start       │  (Bóc tách môn, tuần, loại câu hỏi)
              └─────────────┬─────────────┘
                            │
                            ▼
              ┌───────────────────────────┐
              │       gate.precheck       │ ──► [ Thiếu / Sai mã môn ] ──► Clarify
              └─────────────┬─────────────┘
                            │ (Môn hợp lệ)
                            ▼
              ┌───────────────────────────┐
              │     retrieve.retrieve     │
              └─────────────┬─────────────┘
             ┌──────────────┴──────────────┐
             ▼                             ▼
     [ Có chỉ định Tuần ]        [ Không nêu Tuần cụ thể ]
       week_scan()                 Hybrid Search (Dense + BM25)
  (Lấy toàn bộ chunk tuần,         + Tái sử dụng normalise_scores()
   xếp theo type chuẩn)            + Reciprocal Rank Fusion (k=60)
             └──────────────┬──────────────┘
                            │
                            ▼
              ┌───────────────────────────┐
              │        gate.decide        │  (Tính Rerank Score vs Ngưỡng)
              └─────────────┬─────────────┘
             ┌──────────────┴──────────────┐
             ▼                             ▼
       [ Điểm < Ngưỡng ]             [ Điểm >= Ngưỡng ]
            Abstain                    answer_stage()
     (Kèm gợi ý tổng quan)           (LLM sinh câu trả lời)
                                           │
                                           ▼
                               ┌───────────────────────┐
                               │    check_citations    │ ──► Sai/Thiếu ──► Nhắc sửa 1 lần
                               └───────────┬───────────┘
                                           │ (Hợp lệ)
                                           ▼
                                [ Gắn Footer nguồn ]
```

### 2.1. Tầng Truy xuất 2 Pha (`retrieve.py`)

Hệ thống phân tách rõ rệt thành hai bài toán tìm kiếm:
1. **Pha 1: Quét tuần cứng (`week_scan`)**
   - **Bối cảnh:** Khi câu hỏi nêu rõ tuần (ví dụ: *"Tuần 2 học gì?"*) hoặc tính toán được tuần tương đối từ mốc bắt đầu học kỳ `semester_start` (ví dụ: *"tuần này học gì"*).
   - **Giải pháp:** Truy xuất toàn bộ các chunk thuộc tuần của môn đó và sắp xếp tuần tự theo thứ tự hiển thị chuẩn: `concept` $\rightarrow$ `activity` $\rightarrow$ `assessment` $\rightarrow$ `reference`.
   - **Hiệu quả:** Recall đạt mức tuyệt đối 100%, không bị ảnh hưởng bởi độ lệch ngữ nghĩa của mô hình embedding.

2. **Pha 2: Tìm kiếm hỗn hợp Hybrid Retrieval (Dense Vector + Sparse BM25)**
   - **Bối cảnh:** Khi câu hỏi không nêu tuần mà hỏi về một khái niệm/kỹ năng trải dài (ví dụ: *"Box Model margin padding gồm những gì?"*).
   - **Giải pháp:**
     - Lọc metadata cứng trên Qdrant (`course_id`, `types`).
     - Truy xuất top ứng viên Dense thông qua khoảng cách Cosine trên không gian vector.
     - Truy xuất top ứng viên Sparse thông qua giải thuật BM25Okapi trên tập âm tiết tiếng Việt.
     - **Tái sử dụng bài học lớp tuần 4 (`week4/ex2_1_hybrid_search.py`):** Tích hợp hàm `normalise_scores(scores)` để chuẩn hoá phân phối điểm BM25 về khoảng $[0, 1]$ trước khi kết hợp.
     - Dung hợp thứ hạng hai kênh bằng thuật toán **RRF (Reciprocal Rank Fusion)** với hằng số $k=60$:
       $$RRF\_Score(d) = \sum_{m \in \{Dense, BM25\}} \frac{1}{60 + rank_m(d)}$$

### 2.2. Bộ kiểm soát Gatekeeper (`gate.py`)

Gatekeeper đóng vai trò là "chốt chặn" phân luồng câu hỏi trước khi tốn token gọi mô hình sinh ngôn ngữ:
1. **Kiểm tra sơ bộ (`precheck`):**
   - Chạy trước khi thực hiện truy xuất.
   - Nếu người dùng chưa cung cấp mã môn $\rightarrow$ chuyển hướng sang `clarify` kèm gợi ý: `"Các môn hiện có: WEB101, WEB201..."`.
   - Nếu mã môn nhập vào không tồn tại trong hệ thống $\rightarrow$ chuyển hướng `clarify` báo môn không hợp lệ.
2. **Xếp hạng lại & So khớp ngưỡng (`rerank` & `decide`):**
   - Điểm số RRF chỉ phản ánh thứ tự xếp hạng tương đối nên không dùng để so ngưỡng trực tiếp.
   - Hàm `rerank()` tính toán điểm số kết hợp ngữ nghĩa và độ phủ từ khóa:
     $$Rerank\_Score = 0.7 \times Cosine\_Score + 0.3 \times Coverage(query, text)$$
   - Hàm `decide()` so sánh `top1_score` với `GATE_THRESHOLD` (mặc định 0.5):
     - Nếu `top1_score < threshold` hoặc không có tài liệu $\rightarrow$ chuyển hướng `abstain` (từ chối trả lời) kèm tóm tắt các tuần học của môn (`course_overview`) để hướng dẫn sinh viên hỏi đúng chủ đề.
     - Nếu `top1_score \ge threshold` $\rightarrow$ chấp thuận chuyển tiếp sang tầng trả lời (`answer`).

### 2.3. Định dạng Prompt & Kiểm định Nguồn (`prompts.py`)

- **An toàn chuỗi:** Sử dụng cơ chế thay thế chuỗi tự viết `fill()` thay cho `str.format()` để tránh lỗi Runtime khi nội dung bài giảng lập trình web chứa các ký tự ngoặc nhọn `{}` trong cú pháp CSS/JS.
- **Dựng bằng chứng (`build_evidence`):** Định dạng chuẩn mực từng chunk đưa vào LLM:
  `[{chunk_id}] {course_id} · Tuần {week} · {topic} · {type} (v{version})\n{text}`
- **Kiểm định trích dẫn (`check_citations`):** Quét toàn bộ mã `[chunk_id]` trong câu trả lời sinh ra. Mọi mã trích dẫn bắt buộc phải nằm trong tập hợp các chunk được cung cấp trong evidence.

### 2.4. Động cơ Điều phối Pipeline (`pipeline.py`)

Hàm `run_question()` xử lý toàn diện một yêu cầu:
1. Tạo đối tượng `RunContext` mới cho mỗi truy vấn (hoàn toàn phi trạng thái, độc lập giữa các phiên).
2. Chạy lần lượt: `start` $\rightarrow$ `precheck` $\rightarrow$ `retrieve` $\rightarrow$ `gate`.
3. Nếu Gatekeeper phán quyết `clarify` hoặc `abstain`, trả về ngay câu trả lời định hướng mà không cần gọi LLM (tiết kiệm 100% token).
4. Nếu Gatekeeper phán quyết `answer`, kích hoạt `answer_stage()`:
   - Gọi LLM sinh câu trả lời dựa trên evidence.
   - Nếu LLM phát hiện không đủ cơ sở và trả về `KHONG_DU_CAN_CU`, hệ thống chuyển trạng thái sang `abstained`.
   - Kiểm tra trích dẫn nguồn. Nếu phát hiện trích dẫn sai hoặc bịa đặt mã chunk, hệ thống tự động gửi yêu cầu sửa đổi (retry) một lần kèm danh sách các ID hợp lệ. Nếu lần thứ hai vẫn sai, hệ thống từ chối trả lời (`abstained`) để bảo vệ tính chính xác.
   - Tự động sinh footer tổng hợp nguồn tài liệu ở cuối câu trả lời (ví dụ: `Nguồn: WEB101 tuần 2 (concept); WEB101 tuần 2 (assessment)`).

### 2.5. Minh hoạ Chi tiết: Một Câu hỏi Mẫu Đi qua Pipeline

Để kiểm chứng toàn bộ thiết kế, dưới đây là luồng xử lý chi tiết từng bước của một câu hỏi mẫu trong thực tế:

* **Câu hỏi đầu vào:** `"Tuần 2 học gì?"`, tham số `--course WEB101`.

1. **Bước 1: Khởi tạo (`start`)**
   - Chuẩn hoá mã môn: `course_id = "WEB101"`.
   - Phân tích văn bản: Tìm thấy mẫu `Tuần 2` $\rightarrow$ gán `week = 2`.
   - Phân loại ý định (`infer_types`): Nhận diện câu hỏi tổng quát $\rightarrow$ `types = ()`.
   - Tạo đối tượng: `Query(text="Tuần 2 học gì?", course_id="WEB101", week=2, mode="hybrid")` và `RunContext(run_id="f8a1c9e2b4d3")`.

2. **Bước 2: Kiểm soát sơ bộ (`gate.precheck`)**
   - Kiểm tra `course_id`: `"WEB101"` tồn tại trong danh mục `index.courses` $\rightarrow$ `precheck` trả về `None` (Cho phép tiếp tục truy xuất).

3. **Bước 3: Truy xuất dữ liệu (`retrieve.retrieve`)**
   - Vì có cả `course_id="WEB101"` và `week=2`, hệ thống tự động kích hoạt chiến lược **`week_scan()`**:
   - Lọc và lấy trọn vẹn các chunk của Tuần 2 môn WEB101 theo thứ tự chuẩn `TYPE_ORDER`:
     1. `WEB101-w02-concept` (Chủ đề: CSS cơ bản - Selector, Box Model).
     2. `WEB101-w02-assessment` (Chủ đề: CSS cơ bản - Bài tập tính tổng/trang trí layout).
   - Đóng gói thành danh sách 2 `SearchHit` với nhãn `via="week_scan"`.

4. **Bước 4: Phán quyết Gatekeeper (`gate.decide`)**
   - Nhận diện `hits[0].via == "week_scan"` $\rightarrow$ Không cần so ngưỡng rerank.
   - Trả về: `GateDecision(action="answer", reason="week_found")`.

5. **Bước 5: Sinh câu trả lời & Kiểm định trích dẫn (`answer_stage`)**
   - **Tập ID cho phép:** `allowed = {"WEB101-w02-concept", "WEB101-w02-assessment"}`.
   - **Evidence xây dựng:**
     ```text
     [WEB101-w02-concept] WEB101 · Tuần 2 · CSS cơ bản · concept (v2026.1)
     CSS cơ bản: Selector (class, id, tag), thuộc tính font, màu sắc, Box Model (margin, padding, border).
     [WEB101-w02-assessment] WEB101 · Tuần 2 · CSS cơ bản · assessment (v2026.1)
     Bài tập 2: Tạo trang thông tin cá nhân và áp dụng CSS Box Model để căn chỉnh bố cục.
     ```
   - **LLM phản hồi:**
     ```text
     Tuần 2 môn WEB101 học về CSS cơ bản bao gồm các bộ chọn Selector (class, id, tag) và mô hình hộp Box Model (margin, padding, border) [WEB101-w02-concept]. Sinh viên thực hành bài tập tạo trang thông tin cá nhân và định dạng bố cục bằng CSS [WEB101-w02-assessment].
     ```
   - **Kiểm tra trích dẫn (`check_citations`):**
     - Các trích dẫn tìm thấy: `['WEB101-w02-concept', 'WEB101-w02-assessment']`.
     - Tập không hợp lệ: `invalid = ()` $\rightarrow$ `CitationCheck.ok = True`.
   - **Sinh Footer nguồn:**
     `Nguồn: WEB101 tuần 2 (assessment); WEB101 tuần 2 (concept)`.

6. **Bước 6: Kết thúc & Lưu vết Telemetry (`_finish`)**
   - Trạng thái `status = "answered"`, `citation_ok = True`.
   - Thống kê `Usage`: ghi nhận `prompt_tokens`, `completion_tokens`, `latency_ms`.
   - Ghi toàn bộ trace sự kiện vào file log JSONL mang `run_id`.

---

## 3. Nhật ký Hiện thực & Tái sử dụng Bài học Lớp

* **Nguồn tái sử dụng:** Bài thực hành tuần 4 (`week4/ex2_1_hybrid_search.py`).
* **Hàm tái sử dụng:** `normalise_scores(scores: list[float]) -> list[float]`
* **Vị trí tích hợp:**
  - Được khai báo tại dòng 91 – 98 trong file [`src/t3/retrieve.py`](file:///d:/aipr1/study-assistant/src/t3/retrieve.py#L91-L98).
  - Được gọi tại dòng 111 trong hàm `retrieve()` để chuẩn hoá miền giá trị của điểm số BM25 trước khi tiến hành dung hợp xếp hạng với vector dense.

---

## 4. Kết quả Kiểm thử & Đảm bảo Chất lượng

Hai file kiểm thử chuyên biệt được thiết kế để bao phủ toàn bộ các trường hợp biên và luồng nghiệp vụ:
1. [`tests/test_retrieve_gate.py`](file:///d:/aipr1/study-assistant/tests/test_retrieve_gate.py):
   - `test_week_from_text`: Bóc tách tuần chính xác từ `"Tuần 2"` hoặc `"tuần này"`.
   - `test_week_scan_returns_whole_week_in_type_order`: Kiểm tra week-scan lấy đủ các mục theo đúng thứ tự ưu tiên.
   - `test_course_filter_is_hard`: Đảm bảo bộ lọc môn học hoạt động tuyệt đối, không lẫn môn khác.
   - `test_hybrid_has_rrf_and_bm25_scores_and_dense_has_none`: Kiểm tra cấu trúc điểm số của Hybrid Search.
   - `test_precheck`: Kiểm tra các nhánh thiếu/sai mã môn học.
   - `test_gate_threshold_controls_answer_vs_abstain`: Kiểm tra tính thuần của hàm phán quyết và khả năng so khớp ngưỡng.
   - `test_gate_abstains_on_out_of_scope`: Kiểm tra từ chối câu hỏi lạc đề (*"Cách nấu phở bò?"*).

2. [`tests/test_pipeline.py`](file:///d:/aipr1/study-assistant/tests/test_pipeline.py):
   - `test_answer_branch_cites_and_logs_usage`: Kiểm tra luồng trả lời thành công kèm trích dẫn và log token.
   - `test_no_llm_branches_still_log_zero_usage`: Nhánh không gọi LLM vẫn có bản ghi log token bằng 0.
   - `test_citation_retry_then_success`: Cơ chế nhắc nhở LLM tự sửa trích dẫn khi bị thiếu.
   - `test_invented_source_is_refused_after_retry`: Từ chối dứt khoát nếu LLM cố tình bịa nguồn sau 2 lần thử.
   - `test_requests_do_not_share_state`: Đảm bảo tính phi trạng thái giữa hai request độc lập.

### Kết quả chạy kiểm thử thực tế:
```bash
$env:PYTHONPATH="src"; pytest tests/test_retrieve_gate.py tests/test_pipeline.py
```
```text
============================= test session starts =============================
platform win32 -- Python 3.14.7, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\aipr1\study-assistant
configfile: pytest.ini
collected 20 items

tests\test_retrieve_gate.py ..........                                   [ 50%]
tests\test_pipeline.py ..........                                        [100%]

============================= 20 passed in 0.12s ==============================
```
**Đánh giá:** Toàn bộ **20/20 test cases** đều đạt kết quả **Pass 100% ("xanh")**, đảm bảo sự ổn định và sẵn sàng cho việc đóng gói và báo cáo nộp bài.
