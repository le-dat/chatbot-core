ivc
│
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py          # Khởi chạy FastAPI, kết nối Route
│   │   ├── api/             # HTTP Rest Endpoints (Pause/Resume endpoints)
│   │   └── agents/          # Hệ thống AI Logic
│   │       ├── __init__.py
│   │       ├── state.py     # Định nghĩa Schema cấu trúc dữ liệu chung
│   │       ├── graph.py     # Lắp ghép Nodes, Edges, và Compile Graph
│   │       └── nodes.py     # Logic xử lý của từng Agent đơn lẻ
│   │
│   ├── requirements.txt
│   └── .env