# Phase 1: Foundation, State & Checkpointer

## 1. Kiến trúc tổng thể (Phần liên quan đến Phase 1)
```
ivc/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── core/
│   │   │   ├── config.py          # Pydantic Settings (env vars)
│   │   │   └── logging.py         # Structured logging (structlog)
│   │   ├── api/
│   │   │   ├── chat.py            # POST /chat  (invoke)
│   │   │   └── approve.py         # POST /chat/approve (HITL resume)
│   │   ├── agents/
│   │   │   ├── state.py           # AgentState + reducers
│   │   │   ├── graph.py           # Graph compile
│   │   │   └── nodes/
│   │   │       ├── research.py
│   │   │       ├── execute.py
│   │   │       └── fallback.py
│   │   └── core/
│   │       └── checkpointer.py    # PostgresSaver checkpointer (mới)
```

---

## 2. LangGraph — State Management & Loop Prevention

### 2.1 State hiện tại — vấn đề và fix

```python
# state.py — HIỆN TẠI (có bug)
loop_count: Annotated[int, lambda old, new: old + new]
# Bug: mỗi node trả về loop_count=1, reducer cộng dồn → 1+1+1+...
# Nhưng SOFT_LOOP_LIMIT so sánh với giá trị tích lũy → đúng logic
# Vấn đề: loop_count không bao giờ reset khi conversation mới
```

```python
# state.py — SAU REFACTOR
import operator
from typing import Annotated, TypedDict, Optional
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

# Giới hạn số vòng lặp tối đa để tránh infinite loop
SOFT_LOOP_LIMIT = 4

class AgentState(TypedDict):
    # messages: Lưu trữ lịch sử hội thoại, sử dụng reducer add_messages để append tin nhắn mới
    messages:      Annotated[list[AnyMessage], add_messages]
    # user_query: Câu hỏi ban đầu từ phía người dùng
    user_query:    str
    # current_agent: Tên của agent hiện tại đang xử lý request
    current_agent: str
    # last_agent: Tên của agent xử lý ngay trước đó (dùng để tracking hoặc debug)
    last_agent:    str
    # needs_review: Cờ đánh dấu cần có sự kiểm duyệt của con người (Human-in-the-Loop)
    needs_review:  bool
    # loop_count: Bộ đếm số vòng lặp, sử dụng operator.add để cộng dồn giá trị qua các node
    loop_count:    Annotated[int, operator.add]
    # context_docs: Danh sách các tài liệu tìm kiếm được từ RAG dùng làm ngữ cảnh
    context_docs:  list[str]
    # intent: Ý định của người dùng sau khi được phân loại (ví dụ: chuyển tiền, tìm kiếm thông tin)
    intent:        Optional[str]
    # error: Lưu trữ và lan truyền lỗi nếu có sự cố xảy ra trong graph
    error:         Optional[str]
```

### 2.2 Cách agent truyền tin cho nhau qua State

**Cơ chế:** LangGraph dùng **reducer functions** để merge state updates.
```
Node A trả về dict  →  Reducer merge vào State  →  Node B đọc State
```
- `messages`: dùng `add_messages` reducer → **append**, không replace
- `loop_count`: dùng `operator.add` → **cộng dồn** qua các vòng
- Các field khác: **replace** (last-write-wins)

### 2.3 Chống Infinite Loop — 3 lớp bảo vệ

```python
# graph.py — sau refactor
# LỚP 1: Soft limit — dùng fallback node để chuyển hướng xử lý khi vượt ngưỡng vòng lặp
SOFT_LOOP_LIMIT = 4

# Hàm kiểm tra xem đã chạm tới giới hạn số vòng lặp tối đa hay chưa
def _is_loop_limit_reached(state: AgentState) -> bool:
    return state.get("loop_count", 0) >= SOFT_LOOP_LIMIT

# LỚP 2: LangGraph built-in recursion limit (Cấu hình giới hạn đệ quy mặc định của LangGraph để tự ngắt)
config = {"configurable": {"thread_id": tid}, "recursion_limit": 10}

# LỚP 3: Intent-based routing thay vì keyword matching (Định tuyến thông minh dựa trên intent của State)
def research_router(state: AgentState) -> str:
    # Nếu vượt quá giới hạn vòng lặp, định tuyến sang agent fallback để giải quyết lỗi
    if _is_loop_limit_reached(state):
        return "fallback"
    
    # Định tuyến dựa trên ý định (intent) được phân tích từ tin nhắn
    intent = state.get("intent", "")
    if intent == "execute_transfer":
        return "execute"
    
    return "end"

# Thêm cạnh nối từ fallback_agent trực tiếp tới END để tránh lặp lại chu kỳ lỗi
workflow.add_edge("fallback_agent", END)
```

---

## 3. Human-in-the-Loop Gate (FastAPI + LangGraph)

### 3.1 Cơ chế hoạt động
```
POST /chat  →  graph.invoke()  →  execute_node gọi interrupt()
                                         ↓
                               Graph PAUSES, serialize state vào Checkpointer
                                         ↓
                               API trả về {is_paused: true, pending_approval: {...}}
                                         ↓
                         Frontend hiển thị dialog xác nhận cho user
                                         ↓
POST /chat/approve {approved: true/false}  →  graph.invoke(Command(resume=approved))
                                         ↓
                               Graph RESUME từ điểm interrupt, tiếp tục execute_node
```

### 3.2 Implementation chi tiết
```python
# nodes/execute.py — sau refactor
from langgraph.types import interrupt
from app.agents.state import AgentState

# Node thực thi giao dịch tài chính, có tích hợp cơ chế phê duyệt thủ công (HITL)
async def execute_agent_node(state: AgentState) -> dict:
    # Tạm dừng luồng chạy (interrupt) và gửi yêu cầu phê duyệt về phía Client
    approved: bool = interrupt({
        "type": "approval_required", # Loại ngắt để frontend nhận diện và render dialog phù hợp
        "message": "Xác nhận thực thi giao dịch?",
        "payload": {
            "user_query": state.get("user_query"),
            "amount": state.get("intent_payload", {}).get("amount"),
            "recipient": state.get("intent_payload", {}).get("recipient"),
        }
    })

    # Nếu người dùng từ chối giao dịch
    if not approved:
        return {
            "messages": [AIMessage(content="Giao dịch bị từ chối.")],
            "loop_count": 1, # Tăng số loop_count để kiểm soát giới hạn vòng lặp
        }
    
    # Nếu được phê duyệt, tiến hành thực hiện giao dịch thực tế
    result = await _perform_transaction(state)
    return {
        "messages": [AIMessage(content=f"Giao dịch thành công: {result}")],
        "loop_count": 1,
        "needs_review": False, # Giao dịch đã được duyệt xong, tắt cờ review
    }
```

```python
# core/checkpointer.py
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from app.core.config import settings

async def get_checkpointer():
    """
    Thay thế MemorySaver (in-memory) bằng PostgresSaver để lưu trữ/khôi phục trạng thái (state) bền vững qua các lần khởi động lại hệ thống
    """
    return await AsyncPostgresSaver.from_conn_string(
        settings.DATABASE_URL # Lấy chuỗi kết nối database từ cấu hình hệ thống
    )
```

---

## 4. Kế hoạch triển khai Step-by-Step (Phase 1)

*   **[TEST FIRST]** `[ ]` **Step 1.1:** Tạo file test [test_agents_state.py](file:///home/verno/Desktop/Projects/personal/learn/ivc/backend/tests/test_agents_state.py) để kiểm tra:
    *   `loop_count` reducer hoạt động đúng với `operator.add` (cộng dồn và reset trong fallback).
    *   Cấu trúc của `AgentState` chứa các field mới (`context_docs`, `intent`, `error`).
    *   Hành vi của `_is_loop_limit_reached` hoạt động chuẩn xác.
*   **[TEST FIRST]** `[ ]` **Step 1.2:** Tạo file test [test_checkpointer.py](file:///home/verno/Desktop/Projects/personal/learn/ivc/backend/tests/test_checkpointer.py) để test tính năng persist trạng thái và `interrupt` / `resume` (sử dụng AsyncPostgresSaver hoặc mock in-memory).
*   **[SOURCE]** `[ ]` **Step 1.3:** Tạo file [config.py](file:///home/verno/Desktop/Projects/personal/learn/ivc/backend/app/core/config.py) định nghĩa cấu hình hệ thống bằng Pydantic Settings.
*   **[SOURCE]** `[ ]` **Step 1.4:** Cập nhật [state.py](file:///home/verno/Desktop/Projects/personal/learn/ivc/backend/app/agents/state.py) thay thế reducer của `loop_count` bằng `operator.add` và thêm các trường mới.
*   **[SOURCE]** `[ ]` **Step 1.5:** Thay thế `MemorySaver` bằng `PostgresSaver` trong [graph.py](file:///home/verno/Desktop/Projects/personal/learn/ivc/backend/app/agents/graph.py).
*   **[SOURCE]** `[ ]` **Step 1.6:** Di chuyển các node logic ra thư mục [nodes](file:///home/verno/Desktop/Projects/personal/learn/ivc/backend/app/agents/nodes/) riêng biệt (ví dụ: `research.py`, `execute.py`, `fallback.py`).
*   **[VERIFY]** `[ ]` **Step 1.7:** Chạy `pytest backend/tests/test_agents_state.py` và `test_checkpointer.py` đạt 100% Passed.
