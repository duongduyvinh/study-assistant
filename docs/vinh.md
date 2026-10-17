# Nội dung Báo cáo Phần Phụ trách của Duy Vinh (Ghép vào `report.docx`)

- **Người thực hiện:** Dương Duy Vinh
- **Các mục đảm nhiệm trong `report.docx`:**
  - **Mục 2 (Nửa sau — 0,5 trang):** Luồng xử lý 1 câu hỏi mẫu đi qua Pipeline (`pipeline.py`).
  - **Mục 3 (Nửa sau — 0,75 trang):** Thuật toán Hybrid Search (`retrieve.py`), Cơ chế Gate & Chọn ngưỡng từ bảng Sweep (`gate.py`), Kiểm soát trích dẫn (`prompts.py`).
  - **Mục 6 (Phân tích lỗi — 0,4 trang):** Phân tích sâu 1 ca thất bại (từ chối nhầm `q04` khi thay đổi `GATE_THRESHOLD`).
- **Các file code phụ trách:** `src/t3/retrieve.py`, `src/t3/gate.py`, `src/t3/pipeline.py`, `src/t3/prompts.py`, `prompts/system.txt`, `prompts/answer.txt`, `tests/test_retrieve_gate.py`, `tests/test_pipeline.py`.

---

## MỤC 2 (NỬA SAU). LUỒNG XỬ LÝ MỘT CÂU HỎI MẪU QUA PIPELINE (`pipeline.py`)

Xét câu hỏi mẫu: **`"Tuần 2 học gì?"`** kèm tham số **`--course WEB101`**, được điều phối qua hàm `run_question()` trong `src/t3/pipeline.py` theo 6 bước phi trạng thái (mỗi câu hỏi khởi tạo một `RunContext` riêng):

```text
Input ─► ① start ─► ② precheck ─► ③ retrieve ─► ④ gate.decide ─► ⑤ answer_stage ─► ⑥ _finish + log
                      ├─ clarify                     ├─ abstain       └─ abstain (nếu trích sai 2 lần)
```

1. **Khởi tạo (`pipeline.start`):** Chuẩn hoá `course_id="WEB101"`, dùng biểu thức chính quy trong `week_from_text()` tách số tuần `week=2` (hỗ trợ cả `"tuần 2"` lẫn `"tuần này"` tính từ `semester_start`), nhận diện câu hỏi tổng quát `types=()` qua `infer_types()` và đóng gói vào `Query`, `Filters`, `RunContext`.
2. **Tiền kiểm (`gate.precheck`):** Kiểm tra `"WEB101"` tồn tại trong `index.courses` nên trả về `None` để đi tiếp *(nếu thiếu hoặc sai mã môn, hệ thống lập tức trả về nhánh **`clarify`** kèm danh sách môn hiện có và dừng sớm, tốn 0 token LLM)*.
3. **Truy xuất ngữ cảnh (`retrieve.py`):** Do bộ lọc có cả `course_id="WEB101"` và `week=2`, hệ thống kích hoạt `week_scan()`, lấy trọn vẹn 2 chunk của tuần 2 theo thứ tự sư phạm `TYPE_ORDER`: `WEB101-w02-concept` (Selector, Box Model) và `WEB101-w02-assessment` (Trang trí CV bằng CSS) với nhãn `via="week_scan"` *(nếu câu hỏi không nêu tuần, hệ thống mới chạy tìm kiếm `hybrid` kết hợp Dense + BM25)*.
4. **Cổng gác 3 nhánh (`gate.decide`):** Xét duyệt theo 3 nhánh (`answer`, `abstain`, `clarify`). Vì kết quả đến từ `week_scan` của tuần hợp lệ, hàm duyệt thẳng `GateDecision(action="answer", reason="week_found")` *(nếu đi từ nhánh hybrid mà điểm tái xếp hạng thấp hơn `GATE_THRESHOLD` hoặc tuần không có dữ liệu, hệ thống sẽ rẽ sang **`abstain`** kèm gợi ý outline môn học mà không gọi LLM)*.
5. **Nạp Prompt, sinh câu trả lời & kiểm định trích dẫn (`answer_stage`):** Hai chunk được `build_evidence()` đóng gói vào thẻ `<evidence>` với tập ID hợp lệ `allowed = {"WEB101-w02-concept", "WEB101-w02-assessment"}`, nạp vào `prompts/system.txt` (chỉ dùng evidence, coi evidence là dữ liệu, không bịa nguồn) và `prompts/answer.txt` (mỗi ý kèm `[chunk_id]`, thiếu căn cứ trả `KHONG_DU_CAN_CU`). Câu trả lời của LLM được `check_citations()` quét mã `[chunk_id]` để đối chiếu với tập `allowed` (`invalid = ()` $\rightarrow$ `citation_ok=True`) và nối dòng `Nguồn: WEB101 tuần 2 (assessment); WEB101 tuần 2 (concept)` *(nếu LLM trích dẫn thiếu hoặc bịa ID ngoài `allowed`, hệ thống nhắc sửa 1 lần qua `citation_retry`; nếu lần 2 vẫn sai sẽ ép về **`abstain`**)*.
6. **Kết thúc & Ghi vết (`_finish` & `logging_jsonl.py`):** Gán `status="answered"`, thống kê `Usage` và ghi toàn bộ sự kiện ra file `logs/*.jsonl` theo `run_id`.

---

## MỤC 3 (NỬA SAU). THIẾT KẾ HYBRID SEARCH, CƠ CHẾ GATE & TRÍCH DẪN

### 3.3. Tìm kiếm Hỗn hợp (Hybrid Search) & Tái sử dụng Bài thực hành Tuần 4 (`retrieve.py`)

Khi câu hỏi không chỉ định số tuần cụ thể (ví dụ: *"Box model gồm những gì?"* hoặc *"Closure là gì?"*), hệ thống không thể quét theo tuần mà thực hiện **Hybrid Search** kết hợp hai kênh tìm kiếm bổ trợ nhau:

* **Lọc cứng theo Metadata (`_qdrant_filter` & `matches`):** Trước khi xếp hạng, mọi ứng viên trên cả Qdrant và BM25 đều bị lọc cứng theo `course_id` và `types`, đảm bảo các môn có chủ đề chồng lấn (như JavaScript ở `WEB101` và `WEB201`) không bao giờ lấy nhầm tài liệu của nhau.
* **Kênh Ngữ nghĩa (Dense Search):** Nhúng câu hỏi qua `embedder.embed_query()` và truy vấn khoảng cách Cosine trên Qdrant in-memory (`_dense_top`) lấy tập ứng viên `candidates = max(k * 3, 10)`.
* **Kênh Từ khoá (Lexical Search – BM25):** Tách âm tiết unigram + bigram tiếng Việt (`text.tokenize`) và tính điểm `BM25Okapi` (`_bm25_top`), giúp bắt chính xác các thuật ngữ kỹ thuật đặc thù (`h1-h6`, `if-else`, `margin`, `padding`, `DOM`).
* **Tái sử dụng từ `w4/ex2_1_hybrid_search.py`:**
  - Kế thừa trực tiếp hàm **`normalise_scores(scores: list[float]) -> list[float]`** (khai báo tại dòng 91–100 và gọi tại dòng 122–123 trong `src/t3/retrieve.py`) để chuẩn hoá miền điểm BM25 thô (vốn không có cận trên) về đoạn $[0, 1]$ theo công thức $(s - s_{\min}) / (s_{\max} - s_{\min})$, lưu vào `SearchHit.bm25_score` và làm tiêu chí phân định thứ hạng phụ (tie-breaker).
  - Hợp nhất hai bảng xếp hạng Dense và BM25 bằng thuật toán **Reciprocal Rank Fusion (RRF)** với hằng số $k = 60$ (dòng 125–131 trong `retrieve.py`):
    $$\text{RRF\_Score}(d) = \frac{1}{60 + \text{rank}_{\text{dense}}(d)} + \frac{1}{60 + \text{rank}_{\text{bm25}}(d)}$$

### 3.4. Cơ chế Cổng gác (`gate.py`) & Chọn ngưỡng `GATE_THRESHOLD` từ Bảng Sweep

Điểm RRF chỉ phụ thuộc vào thứ hạng tương đối ($1/(60+1) \approx 0.0164$ ngay cả với câu hỏi lạc đề hoàn toàn), nên không thể dùng RRF để chặn câu hỏi ngoài chương trình. Do đó, hàm thuần `gate.decide()` thực hiện bước tái xếp hạng (`rerank`) bằng tổ hợp tuyến tính giữa độ tương đồng ngữ nghĩa và độ phủ từ khoá nội dung:

$$\text{rerank\_score} = 0.7 \times \text{dense\_score} + 0.3 \times \text{coverage}(\text{query}, \text{chunk})$$

Trong đó `coverage` (trong `text.py`) tính tỷ lệ âm tiết nội dung của câu hỏi (sau khi loại bỏ `STOPWORDS` tiếng Việt) xuất hiện thực tế trong `chunk.text` và `chunk.topic`.

**Lý do chọn ngưỡng dựa trên bảng quét `eval_out/sweep.csv`:**

> *Lưu ý trung thực về dữ liệu:* Bảng `eval_out/sweep.csv` hiện có trong repo được trích xuất từ lần chạy kiểm thử offline (`EMBEDDER=fake` trên 14 câu gốc). Khi chạy với `GEMINI_API_KEY` thật, quy trình chọn ngưỡng giữ nguyên theo bảng quét dưới đây:

| Ngưỡng `GATE_THRESHOLD` | `abstain_precision` | `abstain_recall` | `false_refusal_rate` (Từ chối nhầm) | Đánh giá trên tập Testset |
| :---: | :---: | :---: | :---: | :--- |
| `0.10` | `1.00` | `0.50` | `0.00` | Ngưỡng quá thấp, bỏ lọt 50% câu ngoài chương trình (`no_answer`). |
| **`0.15 – 0.30`** | **`1.00`** | **`1.00`** | **`0.00`** | **Điểm cân bằng tối ưu:** Chặn đúng 100% câu `no_answer`/`near_miss`, không từ chối nhầm câu nào. |
| `0.35 – 0.50` | `0.86` | `1.00` | `0.17` | Ngưỡng mặc định `0.50` quá cao, làm từ chối nhầm 1/6 câu hỏi hợp lệ (`q04` có `top1 = 0.3058`). |
| `0.55 – 0.65` | `0.67` | `1.00` | `0.50` | Từ chối nhầm 50% câu hỏi hợp lệ. |

Từ bảng quét thực nghiệm trên, việc giữ cứng ngưỡng mặc định `0.50` gây ra tỷ lệ từ chối nhầm `16.7%` (câu `q04` chỉ đạt `rerank_score = 0.3058`). Vì toàn bộ các câu `no_answer` và `near_miss` đều có điểm $\le 0.1449$, việc hạ `GATE_THRESHOLD` về vùng **`0.25 – 0.30`** giúp hệ thống đạt đồng thời `abstain_precision = 1.0`, `abstain_recall = 1.0` và `false_refusal_rate = 0.0`.

---

## MỤC 6 (ĐÓNG GÓP CỦA VINH). PHÂN TÍCH SÂU MỘT CA THẤT BẠI (FAILURE CASE ANALYSIS)

Dựa trên kết quả thực tế trong `eval_out/results.csv` (dòng 5 và dòng 19), nhóm phân tích chuyên sâu ca thất bại xảy ra khi để ngưỡng mặc định `GATE_THRESHOLD = 0.50`:

* **Câu hỏi kiểm thử (`id = q04`):** *"Box model gồm những gì?"* (`course_id = "WEB101"`, nhãn `type = "answerable"`, hành vi kỳ vọng: `expected = "answer"`).
* **Kết quả thực tế ở ngưỡng `0.50`:**
  - **Bước Truy xuất (`retrieve.py`):** Cả hai chế độ `dense` và `hybrid` đều tìm **đúng 100%** tài liệu mục tiêu là chunk `WEB101-w02-concept` (*"Selector, Box Model (margin, padding, border), màu sắc, font chữ"*), đạt `recall@k = 1.0`.
  - **Bước Cổng gác (`gate.py`):** Hệ thống lại phán quyết `action = "abstain"` với `reason = "top1_below_threshold"` (`action_correct = False` — **Từ chối nhầm / False Refusal**).
* **Nguyên nhân gốc rễ (Root Cause đối chiếu trực tiếp từ code):**
  1. **Hạn chế của tập `STOPWORDS` trong `src/t3/text.py` làm giảm điểm `coverage`:**
     Hàm `content_syllables("Box model gồm những gì?")` chỉ loại bỏ được 2 từ `"những"` và `"gì"` (có trong `STOPWORDS`), nhưng **không loại bỏ được động từ `"gồm"`** (chưa có trong `STOPWORDS` ở dòng 10–13 của `text.py`). Do đó, tập âm tiết nội dung của câu hỏi gồm 3 từ `{'box', 'model', 'gồm'}`, nhưng trong chunk `WEB101-w02-concept` chỉ xuất hiện 2 từ `'box'` và `'model'`, khiến độ phủ từ khoá bị tụt xuống:
     $$\text{coverage} = \frac{2}{3} \approx 0.6667 \quad (\text{đóng góp } 0.3 \times \frac{2}{3} = 0.2000)$$
  2. **Sự pha loãng vector trên chunk gộp nhiều khái niệm làm giảm `dense_score`:**
     Chunk `WEB101-w02-concept` chứa toàn bộ các khái niệm của Tuần 2 (tổng cộng 22 âm tiết gồm cả header ngữ cảnh: *Môn, Lập trình web cơ bản, Tuần 2, CSS cơ bản, Selector, margin, padding, border, màu sắc, font chữ...*), trong khi câu truy vấn chỉ trùng 2 âm tiết `'box'` và `'model'`. Do đó điểm tương đồng Cosine chỉ đạt $\text{dense\_score} = 0.1512$ (đóng góp $0.7 \times 0.1512 = 0.1058$).
  3. **Tổng điểm tái xếp hạng bị hụt ngưỡng `0.50`:**
     $$\text{top1\_score} = 0.7 \times 0.1512 + 0.3 \times 0.6667 = 0.1058 + 0.2000 = 0.3058 < 0.50$$
* **Hướng khắc phục rút ra từ thực nghiệm:**
  - **Cách 1 (Về xử lý ngôn ngữ):** Bổ sung các động từ hỏi phổ biến như `"gồm"`, `"như"`, `"nêu"` vào danh sách `STOPWORDS` trong `text.py` để `coverage` của câu `q04` đạt đúng $2/2 = 1.0$.
  - **Cách 2 (Về hiệu chỉnh ngưỡng Gate):** Hạ `GATE_THRESHOLD` từ mức giữ chỗ `0.50` xuống `0.25 – 0.30` theo bảng `sweep.csv` để câu `q04` (`0.3058 >= 0.25`) được trả lời đúng mà vẫn chặn toàn bộ các câu ngoài chương trình ($\le 0.1449$).
