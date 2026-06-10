# Phase 3: SSE Streaming & Hallucination Guard

## 1. Kiến trúc tổng thể (Phần liên quan đến Phase 3)
```
ivc/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── stream.py          # GET  /chat/stream (SSE)
│   │   └── agents/
│   │       └── nodes/
│   │           └── research.py    # Structured output & Grounding check
```

---

## 2. SSE Streaming + AbortSignal

### 2.1 Cơ chế hoạt động end-to-end
```
Frontend                          Backend FastAPI               Anthropic API
   |                                    |                              |
   |  GET /chat/stream                  |                              |
   |  (EventSource / fetch + AbortCtrl) |                              |
   |─────────────────────────────────>  |                              |
   |                                    | stream = anthropic.stream()  |
   |                                    |──────────────────────────>   |
   |  <── data: {"token": "Xin"} ──     |  <── token stream ──────    |
   |  <── data: {"token": "chào"} ──    |                              |
   |                                    |                              |
   |  [User clicks Stop]                |                              |
   |  controller.abort()                |                              |
   |─────── Connection closed ────────> |                              |
   |                                    | request.is_disconnected()    |
   |                                    | → True                       |
   |                                    | stream.close() ──────────>   |
   |                                    | (Anthropic API call stopped) |
```

### 2.2 Backend Implementation
```python
# api/stream.py
import asyncio
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from app.agents.graph import app_graph

router = APIRouter()

# Endpoint xử lý SSE Streaming tin nhắn phản hồi của Agent
@router.get("/chat/stream")
async def stream_chat(request: Request, message: str, thread_id: str):
    # Cấu hình thread_id để định danh phiên hội thoại trong LangGraph checkpointer
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = _build_initial_state(message)

    # Hàm generator tạo ra các event SSE liên tục
    async def event_generator():
        try:
            # Lắng nghe các sự kiện stream dạng async từ LangGraph Graph
            async for event in app_graph.astream_events(
                initial_state, config, version="v2"
            ):
                # Kiểm tra định kỳ xem client đã ngắt kết nối (hoặc bấm dừng) chưa để giải phóng tài nguyên
                if await request.is_disconnected():
                    break

                # Chỉ xử lý các sự kiện sinh ra token từ chat model và có nội dung text hợp lệ
                if (
                    event["event"] == "on_chat_model_stream"
                    and event["data"]["chunk"].content
                ):
                    token = event["data"]["chunk"].content
                    # Trả về dữ liệu dạng SSE format (data: ...)
                    yield f"data: {json.dumps({'token': token})}\n\n"

        except asyncio.CancelledError:
            # Xử lý khi tiến trình stream bị hủy bỏ bởi hệ thống
            pass
        finally:
            # Đảm bảo luôn gửi tín hiệu kết thúc luồng cho client nhận biết
            yield "data: [DONE]\n\n"

    # Trả về response dạng StreamingResponse với header đặc thù của SSE
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache", # Ngăn browser lưu cache luồng stream này
            "X-Accel-Buffering": "no",   # Vô hiệu hóa buffering trên proxy Nginx nếu có để giảm lag
        },
    )
```

---

## 3. Hallucination Mitigation (Chống ảo tưởng)

### 3.1 Các kỹ thuật áp dụng
```python
# agents/nodes/research.py
# System prompt nghiêm ngặt ép LLM chỉ được trả lời trong phạm vi context cung cấp
SYSTEM_PROMPT = """Bạn là trợ lý tài chính của IVC.
QUY TẮC BẮT BUỘC:
1. Chỉ trả lời dựa trên CONTEXT được cung cấp.
2. Nếu không có trong CONTEXT, trả lời: "Tôi không tìm thấy thông tin này trong cơ sở dữ liệu."
3. KHÔNG suy đoán số liệu.
4. Trích dẫn nguồn tài liệu cụ thể.
"""

# Kỹ thuật 1: Đặt Temperature = 0 để giảm thiểu tính sáng tạo tối đa của LLM, tăng tính nhất quán và chính xác
llm = ChatAnthropic(model="claude-3-5-haiku-20241022", temperature=0)

# Kỹ thuật 2: Định nghĩa cấu trúc trả về bắt buộc sử dụng Pydantic để dễ dàng kiểm duyệt và lưu trữ state
class ResearchResponse(BaseModel):
    answer: str                                # Nội dung câu trả lời
    sources: list[str]                         # Danh sách nguồn tài liệu tham chiếu
    confidence: Literal["high", "medium", "low"] # Mức độ tin cậy của câu trả lời
    needs_human_review: bool                   # Cờ đánh dấu cần kiểm duyệt thủ công nếu độ tự tin thấp

# Ràng buộc đầu ra của model theo Schema Pydantic
structured_llm = llm.with_structured_output(ResearchResponse)

# Kỹ thuật 3: Thuật toán Grounding check thủ công, kiểm tra xem câu trả lời có chứa ý nào khớp trực tiếp với tài liệu gốc không
def _check_grounding(answer: str, docs: list[str]) -> bool:
    answer_lower = answer.lower()
    # Duyệt qua các câu trong tài liệu gốc có độ dài > 20 ký tự để so sánh độ tương đồng
    return any(
        sentence.strip() in answer_lower
        for doc in docs
        for sentence in doc.split(".")
        if len(sentence.strip()) > 20
    )
```

---

## 4. Kế hoạch triển khai Step-by-Step (Phase 3)

*   **[TEST FIRST]** `[ ]` **Step 3.1:** Tạo file test [test_chat_stream.py](file:///home/verno/Desktop/Projects/personal/learn/ivc/backend/tests/test_chat_stream.py) để kiểm tra endpoint SSE:
    *   Streaming token trả về đúng format `data: {"token": "..."}`.
    *   Khi ngắt kết nối (Client disconnect), Generator lập tức dừng để tiết kiệm API tokens.
*   **[TEST FIRST]** `[ ]` **Step 3.2:** Tạo file test [test_hallucination_guard.py](file:///home/verno/Desktop/Projects/personal/learn/ivc/backend/tests/test_hallucination_guard.py) kiểm tra:
    *   Structured Output bắt buộc của LLM.
    *   Hàm grounding check so sánh câu trả lời với context.
    *   Cơ chế kích hoạt review khi confidence = "low".
*   **[SOURCE]** `[ ]` **Step 3.3:** Viết [stream.py](file:///home/verno/Desktop/Projects/personal/learn/ivc/backend/app/api/stream.py) xử lý SSE với `astream_events` của LangGraph.
*   **[SOURCE]** `[ ]` **Step 3.4:** Cập nhật node research để cấu hình temperature = 0, Pydantic structured output, grounding check và tự nâng cờ `needs_review` hoặc chuyển hướng fallback.
*   **[VERIFY]** `[ ]` **Step 3.5:** Chạy `pytest backend/tests/test_chat_stream.py` và `test_hallucination_guard.py` đạt 100% Passed.
