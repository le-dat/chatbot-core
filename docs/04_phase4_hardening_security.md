# Phase 4: Production Hardening & Security

## 1. Kiến trúc tổng thể (Phần liên quan đến Phase 4)
```
ivc/
├── backend/
│   ├── app/
│   │   └── core/
│   │       ├── security.py        # API key validation middleware (mới)
│   │       └── limiter.py         # Asyncio Semaphore limiter (mới)
├── docker-compose.yml
├── docker-compose.prod.yml
```

---

## 2. Concurrent Sessions Management

```python
# core/config.py
class Settings(BaseModel):
    # WORKERS: Số lượng tiến trình worker chạy ứng dụng FastAPI
    WORKERS: int = 4
    # MAX_CONCURRENT_LLM_CALLS: Số lượng cuộc gọi API LLM đồng thời tối đa cho phép
    MAX_CONCURRENT_LLM_CALLS: int = 10
```

```python
# core/limiter.py
import asyncio

# Khởi tạo Semaphore giới hạn số lượng request LLM chạy đồng thời để tránh tràn rate limit của API
_llm_semaphore = asyncio.Semaphore(10)

# Hàm bao bọc (Wrapper) chạy coroutine giới hạn số lượng tác vụ LLM đồng thời
async def with_llm_limit(coro):
    """
    Context manager giới hạn tối đa 10 LLM calls đồng thời
    """
    async with _llm_semaphore: # Giữ semaphore khi chạy coroutine và tự giải phóng khi xong
        return await coro
```

---

## 3. Self-hosted Docker Compose — Security & Volumes

### 3.1 Environment Variables
```yaml
# docker-compose.prod.yml
services:
  backend:
    image: ivc-backend:latest
    env_file:
      - .env.prod # Đọc các biến môi trường cấu hình production từ file .env.prod
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY} # Cấu hình khóa API cho Anthropic
    secrets:
      - anthropic_key # Nạp Secret bên ngoài vào container dưới dạng file tại /run/secrets/anthropic_key

secrets:
  anthropic_key:
    external: true # Đánh dấu secret được quản lý bên ngoài Docker Compose (ví dụ: tạo thủ công bằng docker secret create)
```

### 3.2 Volume Management
```yaml
# docker-compose.prod.yml
services:
  postgres:
    image: postgres:16-alpine
    volumes:
      - postgres_data:/var/lib/postgresql/data # Gắn volume lưu trữ cơ sở dữ liệu Postgres bền vững
    environment:
      POSTGRES_DB: ivc_db
      POSTGRES_USER: ivc_user
      POSTGRES_PASSWORD_FILE: /run/secrets/postgres_password # Đọc mật khẩu admin Postgres an toàn từ docker secret
    restart: unless-stopped # Tự khởi động lại container trừ khi bị dừng chủ động

  chromadb:
    image: chromadb/chroma:latest
    volumes:
      - chroma_data:/chroma/chroma # Gắn volume lưu trữ vectorstore ChromaDB bền vững
    environment:
      - CHROMA_SERVER_AUTH_CREDENTIALS=${CHROMA_TOKEN} # Khóa token dùng để xác thực quyền truy cập vào ChromaDB
      - CHROMA_SERVER_AUTH_PROVIDER=token # Cấu hình phương thức xác thực bằng static token
    restart: unless-stopped

volumes:
  postgres_data:
    driver: local
    driver_opts:
      type: none
      device: /data/ivc/postgres # Đường dẫn vật lý trên server để bind mount dữ liệu Postgres
      o: bind
  chroma_data:
    driver: local
    driver_opts:
      type: none
      device: /data/ivc/chroma # Đường dẫn vật lý trên server để bind mount dữ liệu ChromaDB
      o: bind
```

---

## 4. Kế hoạch triển khai Step-by-Step (Phase 4)

*   **[TEST FIRST]** `[ ]` **Step 4.1:** Tạo file test [test_security.py](file:///home/verno/Desktop/Projects/personal/learn/ivc/backend/tests/test_security.py) kiểm tra:
    *   API Key Middleware chặn các request thiếu/sai key.
    *   asyncio Semaphore giới hạn số lượng request LLM đồng thời hoạt động tốt.
*   **[SOURCE]** `[ ]` **Step 4.2:** Viết [security.py](file:///home/verno/Desktop/Projects/personal/learn/ivc/backend/app/core/security.py) (FastAPI API key middleware).
*   **[SOURCE]** `[ ]` **Step 4.3:** Setup Semaphore limit trong [limiter.py](file:///home/verno/Desktop/Projects/personal/learn/ivc/backend/app/core/limiter.py).
*   **[SOURCE]** `[ ]` **Step 4.4:** Tạo file [docker-compose.prod.yml](file:///home/verno/Desktop/Projects/personal/learn/ivc/docker-compose.prod.yml) có cấu hình named volumes và Docker secrets.
*   **[VERIFY]** `[ ]` **Step 4.5:** Chạy toàn bộ test suite và audit checklist bằng `python .agents/scripts/checklist.py .`.
