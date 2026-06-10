# Phase 2: RAG Pipeline & ChromaDB Integration

## 1. Kiến trúc tổng thể (Phần liên quan đến Phase 2)
```
ivc/
├── backend/
│   ├── app/
│   │   ├── agents/
│   │   │   └── nodes/
│   │   │       └── rag.py         # RAG retrieval node (mới)
│   │   └── rag/
│   │       ├── embeddings.py      # Primary + HuggingFace fallback
│   │       ├── chunking.py        # RecursiveCharacterTextSplitter config
│   │       ├── vectorstore.py     # ChromaDB client
│   │       └── ingest.py          # Document ingestion pipeline
```

---

## 2. RAG Pipeline — ChromaDB + Chunking

### 2.1 Chunking Strategy

```python
# rag/chunking.py
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Trả về bộ chia văn bản (Text Splitter) được cấu hình động theo loại tài liệu
def get_text_splitter(doc_type: str = "default") -> RecursiveCharacterTextSplitter:
    """
    Cấu hình chunk size dựa trên loại document:
    - Nhỏ quá (< 200 tokens): context thiếu, câu trả lời rời rạc
    - Lớn quá (> 1000 tokens): noise nhiều, retrieval kém chính xác
    - overlap = 10-20% chunk_size để tránh cắt đứt câu/ý
    """
    # Cấu hình cụ thể cho từng loại tài liệu
    configs = {
        "default": {"chunk_size": 512, "chunk_overlap": 64},       # Mặc định
        "financial_doc": {"chunk_size": 768, "chunk_overlap": 128}, # Tài liệu tài chính (cần chunk lớn hơn để giữ ngữ cảnh số liệu)
        "faq": {"chunk_size": 256, "chunk_overlap": 32},           # Câu hỏi thường gặp (ngắn gọn, chunk nhỏ để tránh loãng thông tin)
    }
    # Lấy cấu hình tương ứng, fallback về default nếu không khớp
    cfg = configs.get(doc_type, configs["default"])
    
    # Khởi tạo RecursiveCharacterTextSplitter chia nhỏ văn bản đệ quy theo dấu câu để giữ nguyên câu từ có nghĩa
    return RecursiveCharacterTextSplitter(
        chunk_size=cfg["chunk_size"],
        chunk_overlap=cfg["chunk_overlap"],
        separators=["\n\n", "\n", "。", ".", " ", ""], # Ưu tiên cắt theo đoạn, sau đó đến dòng, dấu chấm và khoảng trắng
        length_function=len,
        add_start_index=True, # Lưu trữ chỉ mục bắt đầu của mỗi chunk trong tài liệu gốc để tiện truy vết
    )
```

### 2.2 HuggingFace Embedding Fallback

```python
# rag/embeddings.py
import logging
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from app.core.config import settings

logger = logging.getLogger(__name__)

# Hàm khởi tạo và trả về bộ tạo vector hóa (Embeddings Model) tương ứng
def get_embeddings():
    """
    Primary: OpenAI text-embedding-3-small
    Fallback: local HuggingFace model ( offline dev )
    """
    # Nếu không cấu hình OpenAI API Key, tự động chuyển sang HuggingFace model cục bộ
    if not settings.OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY not set, using local HuggingFace embeddings")
        return _get_huggingface_embeddings()
    
    try:
        # Cố gắng khởi tạo OpenAI Embeddings làm lựa chọn chính
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=settings.OPENAI_API_KEY,
        )
        # Thực hiện một truy vấn test nhỏ để xác minh kết nối và API Key hợp lệ
        embeddings.embed_query("test")
        return embeddings
    except Exception as e:
        # Trong trường hợp gọi OpenAI lỗi (mất mạng, hết hạn ngạch...), ghi log và chạy fallback sang HuggingFace
        logger.error(f"OpenAI embeddings failed: {e}. Falling back to HuggingFace.")
        return _get_huggingface_embeddings()

# Hàm cấu hình model local HuggingFace phục vụ offline dev hoặc làm fallback
def _get_huggingface_embeddings():
    """
    Model: paraphrase-multilingual-MiniLM-L12-v2 (Hỗ trợ tiếng Việt, nhẹ ~400MB)
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"}, # Chạy trên CPU để tiết kiệm tài nguyên
        encode_kwargs={"normalize_embeddings": True}, # Chuẩn hóa vector đầu ra về dạng vector đơn vị
    )
```

---

## 3. Kế hoạch triển khai Step-by-Step (Phase 2)

*   **[TEST FIRST]** `[ ]` **Step 2.1:** Tạo file test [test_rag_pipeline.py](file:///home/verno/Desktop/Projects/personal/learn/ivc/backend/tests/test_rag_pipeline.py) để kiểm tra:
    *   Cơ chế chia văn bản (chunking) với `RecursiveCharacterTextSplitter` theo đúng cấu hình cho từng loại văn bản.
    *   Cơ chế fallback từ OpenAI Embeddings sang local HuggingFace Embeddings khi lỗi API key hoặc mất kết nối.
    *   Đọc/ghi vector và query tương đồng trên ChromaDB Client.
*   **[SOURCE]** `[ ]` **Step 2.2:** Viết [embeddings.py](file:///home/verno/Desktop/Projects/personal/learn/ivc/backend/app/rag/embeddings.py) cấu hình chính + HuggingFace fallback.
*   **[SOURCE]** `[ ]` **Step 2.3:** Viết [chunking.py](file:///home/verno/Desktop/Projects/personal/learn/ivc/backend/app/rag/chunking.py) và [vectorstore.py](file:///home/verno/Desktop/Projects/personal/learn/ivc/backend/app/rag/vectorstore.py).
*   **[SOURCE]** `[ ]` **Step 2.4:** Viết [rag.py](file:///home/verno/Desktop/Projects/personal/learn/ivc/backend/app/agents/nodes/rag.py) làm retrieval node để tích hợp trực tiếp vào Graph.
*   **[VERIFY]** `[ ]` **Step 2.5:** Chạy `pytest backend/tests/test_rag_pipeline.py` đạt 100% Passed.
