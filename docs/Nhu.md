# Bản báo cáo Nhiệm vụ: LLM Client, Agent Loop, MCP & Logging 

- **Sinh viên:** Thái Ngọc Tâm Như - 24110171
- **Phụ trách các nội dung:** 
+ HTTP/LLM Client (`http.py`, `llm.py`) 
+ Agent loop & Tool Verify (`agent.py`, `tools.py`, `notes.py`)
+ MCP protocol (`mcp_server.py`, `mcp_client.py`)
+ Logging JSONL (`logging_jsonl.py`, `cli.py`) 
+ Testing (`test_llm_http.py`, `test_tools_agent.py`, `test_mcp.py`).

---

## 1. Thiết kế LLM Client & Cơ chế Chịu lỗi (`http.py`, `llm.py`)
- Ở phần này em thực hiện việc gọi LLM về một hàm duy nhất là `call_llm` để kết nối tới endpoint tương thích chuẩn OpenAI (em dùng `gemini-3.5-flash-lite`) 
cho cả nhánh `ask` với `agent`.
- Trong lúc chạy, em gọi API ngoài thường gặp phải các lỗi mạng hoặc giới hạn băng thông, nên em thiết kế cơ chế chịu lỗi ở tầng HTTP như sau:
+ Yêu cầu Retry kèm thời gian chờ tăng lên đối với các mã lỗi tạm thời (`429`, `500`, `502`, `503`, `504`) và lỗi kết nối mạng; báo lỗi ngay khi gặp các mã lỗi cấu hình/xác thực (`400`, `401`, `404`).
+ Em viết thêm hàm `strip_think()` để loại bỏ khối `<think>...</think>` trước khi trích xuất JSON hoặc kiểm tra trích dẫn nhờ đó việc đọc JSON của Agent hay bước kiểm tra trích dẫn ở sau không bị lỗi định dạng.
+ Mỗi lệnh gọi trả về đối tượng `Usage` ghi nhận đầy đủ: `prompt_tokens`, `completion_tokens`, `calls=1`, `max_prompt=prompt_tokens`, và `latency_ms`.

---

## 2. Agent loop, Tool note và cơ chế Verify bắt buộc (`agent.py`, `tools.py`, `notes.py`)
- `notes.py` để lưu và đọc ghi chú của sinh viên.
- `tools.py` để đóng gói các chức năng thành tool để Agent gọi. 
- `agent.py` chứa vòng lặp `run_agent` để tự quản lý danh sách `messages` trong phạm vi một `RunContext`, giới hạn tối đa `MAX_STEPS=6` để không lặp vô hạn.
+ Đối với tool có nhận course_id, Agent buộc phải gán args["course_id"] = mã môn mà user input, không được tự ý thay đổi.
+ Verify bắt buộc sau khi ghi (`save_study_note` -> `list_study_notes`): nếu LLM định trả final mà chưa kiểm tra thì Agent block lại và yêu cầu gọi `list_study_notes`. Khi danh sách có chứa `note_id` đó thì verify được ghi nhận ok. 

---

## 3. MCP Server & Client (`mcp_server.py`, `mcp_client.py`)
Em thêm MCP hoạt động qua `stdio` để tách server và client:
- Giữ sạch luồng `stdout`: Trong giao thức MCP qua `stdio`, `stdout` là đường truyền bản tin JSON-RPC. Do đó, trong `mcp_server.py` và các file liên quan, em hoàn toàn không dùng `print()` mà cấu hình `logging.basicConfig(stream=sys.stderr)` để toàn bộ log chẩn đoán đi qua `stderr`.
- Tool động (`list_tools`): `MCPBackend` khởi chạy phiên kết nối bất đồng bộ trong một luồng nền (daemon thread), gọi `session.list_tools()` lúc khởi tạo để dựng danh sách `ToolDescriptor` động từ JSON Schema của server mà không hardcode tên tool.
- Đồng nhất giao diện `ToolBackend`: Cả `ToolBox` (gọi hàm Python nội bộ) và `MCPBackend` (gọi qua giao thức MCP) đều tuân thủ giao thức `ToolBackend` (`specs()` và `call()`), giúp `run_agent` chuyển đổi trong suốt qua tham số `--backend local|mcp`.

---

## 4. Logging JSONL & Thực nghiệm trên API thật (`logging_jsonl.py`)
- Mỗi phiên chạy (`ask` hoặc `agent`) được gắn một `run_id` duy nhất và xuất thành một file trong `logs/`.
- Cuối mỗi phiên, hàm `write_run_log` tổng hợp toàn bộ `ctx.usages` thành sự kiện `usage_summary` gồm 4 thông số: `calls`, `prompt_tokens`, `completion_tokens`, và `max_prompt`.
- Em thử nghiệm 3 trường hợp bằng cách chạy API thật với gemini-3.5-flash-lite và gemini-embedding-001. Kết quả ghi trong logs/:

| File log | Câu lệnh | Luồng xử lý | Số liệu |
| :--- | :--- | :--- | :--- |
| `ask-20261017-212827-88098a8d6970.jsonl` | *"Tuần 2 học gì?"* (`WEB101`) | Nhánh `answer`, `citation_ok=true` | `calls=1`, `prompt_tokens=307`, `completion_tokens=45`, `max_prompt=307` |
| `ask-20261017-212829-63de3273c537.jsonl` | *"Tuần 2 học gì?"* (thiếu mã môn) | Nhánh `clarify`, không gọi LLM | `calls=0`, `total_tokens=0` |
| `agent-20261017-212836-72da6ddd2339.jsonl` | *"Ghi chú tuần 2 WEB101: ôn Selector và Box Model"* | 3 bước: `save_study_note` → `list_study_notes` (`verify ok=true`) → `final` | `calls=3`, `prompt_tokens=1917`, `completion_tokens=113`, `max_prompt=721` |

---

## 5. Hai hạn chế thực tế của hệ thống
1. **Ngưỡng `GATE_THRESHOLD = 0.5` đang là giá trị mặc định:** Việc chọn ngưỡng tối ưu phụ thuộc vào phân bố điểm tương đồng trên bộ câu hỏi thực tế với embedding thật; khi mở rộng thêm môn học mới cần quét lại bảng `sweep.csv` để hiệu chỉnh ngưỡng tránh từ chối nhầm.
2. **Độ trễ và chi phí token của vòng lặp Agent tăng theo số bước:** Trong tác vụ ghi chú có bước verify bắt buộc, toàn bộ lịch sử hội thoại và kết quả tool được gửi lại qua mỗi bước khiến `prompt_tokens` tăng dần (553 -> 643 -> 721 token, tổng cộng 1917 prompt token cho 3 lần gọi).

