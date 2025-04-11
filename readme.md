# AI Train - RAG Document Retrieval System

## Tổng quan dự án

AI Train là một hệ thống truy vấn tài liệu thông minh sử dụng kiến trúc RAG (Retrieval Augmented Generation). Hệ thống cho phép người dùng truy xuất thông tin từ các tài liệu đã được nhập vào cơ sở dữ liệu bằng cách sử dụng truy vấn bằng ngôn ngữ tự nhiên. Dự án sử dụng các mô hình ngôn ngữ lớn (LLMs) để tạo câu trả lời chính xác và phù hợp với ngữ cảnh dựa trên nội dung của tài liệu.

## Mục đích dự án

Dự án này được phát triển nhằm:
- Xây dựng một hệ thống truy xuất thông tin thông minh từ tài liệu
- Kết hợp kỹ thuật RAG (Retrieval Augmented Generation) để tăng độ chính xác cho câu trả lời
- Hỗ trợ xử lý và tìm kiếm thông tin từ các tài liệu Markdown và PDF
- Cung cấp giao diện API cho việc tích hợp với các ứng dụng khác
- Tối ưu hóa trải nghiệm người dùng trong việc truy vấn thông tin từ kho tài liệu lớn

## Các tính năng chính

- **Xử lý tài liệu**: Trích xuất văn bản từ các file Markdown và PDF
- **Phân đoạn văn bản**: Chia nhỏ văn bản thành các đoạn có kích thước phù hợp
- **Tạo embedding**: Chuyển đổi đoạn văn bản thành vector embedding sử dụng mô hình ngôn ngữ
- **Lưu trữ vector**: Sử dụng ChromaDB để lưu trữ và truy vấn các vector embedding
- **Tìm kiếm ngữ nghĩa**: Tìm kiếm các đoạn văn bản liên quan dựa trên độ tương đồng ngữ nghĩa
- **Tạo câu trả lời thông minh**: Sử dụng mô hình LLM để tạo câu trả lời dựa trên ngữ cảnh tài liệu
- **API REST**: Cung cấp API để tương tác với hệ thống

## Yêu cầu hệ thống

- Python 3.7+
- Ít nhất 8GB RAM
- Kết nối internet (để tải mô hình lần đầu)
- Docker (tùy chọn, nếu triển khai bằng container)

## Cài đặt

### Cài đặt thông thường

1. Clone dự án:
```bash
git clone https://github.com/your-username/ai-train.git
cd ai-train
```

2. Tạo và kích hoạt môi trường ảo:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Cài đặt các phụ thuộc:
```bash
pip install -r requirements.txt
```

4. Kiểm tra cài đặt:
```bash
python main.py ping
```

### Cài đặt với Docker

1. Xây dựng container:
```bash
docker build -t ai-train .
```

2. Chạy container:
```bash
docker run -p 5000:5000 -v ./data:/app/data -v ./chroma_storage:/app/chroma_storage ai-train
```

## Cấu trúc dự án

```
d:\ComputerVision\ai-train\
├── data/                  # Thư mục chứa dữ liệu tài liệu
│   ├── markdown/          # Tài liệu dạng Markdown
│   └── pdf/               # Tài liệu dạng PDF
├── chroma_storage/        # Dữ liệu vector embeddings
├── models/                # Lưu trữ các mô hình đã tải về
├── ingest/                # Module xử lý và nhập liệu
│   ├── embedder.py        # Xử lý việc tạo embedding vector
│   ├── text_extractor.py  # Trích xuất nội dung từ tài liệu
│   └── text_chunker.py    # Chia văn bản thành đoạn nhỏ
├── utils/                 # Các tiện ích
│   ├── file_utils.py      # Xử lý file
│   └── logging_config.py  # Cấu hình logging
├── main.py                # Điểm vào chính của ứng dụng
├── requirements.txt       # Danh sách các phụ thuộc
└── README.md              # Tài liệu hướng dẫn
```

## Hướng dẫn sử dụng

### Chuẩn bị dữ liệu

1. Đặt các file Markdown hoặc PDF vào thư mục `data/`
2. Chạy API để nhập dữ liệu:
```bash
curl http://localhost:5000/ingest
```

### Truy vấn thông tin

Sử dụng API `/search` để tìm kiếm đoạn văn bản liên quan:
```bash
curl "http://localhost:5000/search?query=your search query here"
```

Sử dụng API `/rag` để tạo câu trả lời dựa trên tài liệu:
```bash
curl -X POST http://localhost:5000/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "your question here"}'
```

## API Reference

| Endpoint  | Method | Description                                        |
| --------- | ------ | -------------------------------------------------- |
| `/ping`   | GET    | Kiểm tra trạng thái hoạt động của hệ thống         |
| `/ingest` | GET    | Nhập dữ liệu từ các tài liệu trong thư mục `data/` |
| `/search` | GET    | Tìm kiếm các đoạn văn bản tương tự với truy vấn    |
| `/rag`    | POST   | Tạo câu trả lời dựa trên nội dung tài liệu         |

### Các tham số API

#### `/search`
- `query` (string): Câu truy vấn tìm kiếm

#### `/rag`
- Body JSON:
  ```json
  {
    "query": "Câu hỏi của bạn"
  }
  ```

## Quy trình hoạt động

1. **Nhập dữ liệu**: Trích xuất văn bản từ tài liệu và tạo embedding
2. **Tìm kiếm**: Khi nhận truy vấn, hệ thống tạo embedding cho truy vấn và tìm kiếm các đoạn văn bản tương tự
3. **Tạo câu trả lời**: Hệ thống sử dụng các đoạn văn bản tìm được làm ngữ cảnh để tạo câu trả lời
4. **Tinh chỉnh câu trả lời**: Sử dụng mô hình ngôn ngữ thứ hai để tối ưu và trình bày câu trả lời

## Triển khai

Hệ thống hiện đang sử dụng Ollama API cho các mô hình LLM:
- MODEL_DEEPSEEK = 'deepseek-r1:8b' (phân tích ban đầu)
- MODEL_LLAMA = 'llama3.2' (tinh chỉnh câu trả lời)

Server Ollama được cấu hình tại: `OLLAMA_SERVER = 'http://10.6.18.2:11434/api/generate'`

## Phát triển

Để phát triển thêm tính năng cho dự án, bạn có thể:
1. Fork dự án
2. Tạo nhánh tính năng (`git checkout -b feature/your-feature`)
3. Commit các thay đổi (`git commit -m 'Add some feature'`)
4. Push lên nhánh (`git push origin feature/your-feature`)
5. Tạo Pull Request

## Giấy phép

Dự án này được phân phối theo giấy phép MIT.
