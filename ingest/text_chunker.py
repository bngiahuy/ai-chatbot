import logging
from typing import List
from langchain.text_splitter import MarkdownHeaderTextSplitter, TokenTextSplitter
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

def split_text(
    text: str,
    chunk_size: int = 1024,
    chunk_overlap: int = 100,
    verbose: bool = False
) -> List[str]:
    """
    Tách nội dung Markdown thành các đoạn ngữ nghĩa tối ưu cho embedding E5.
    
    - Ưu tiên tách theo heading
    - Dùng tokenizer của mô hình để chia theo token
    - Mỗi chunk ~350 tokens, có overlap để giữ ngữ cảnh liền mạch
    
    Args:
        text: Nội dung markdown đã xử lý.
        chunk_size: Số lượng token mỗi chunk.
        chunk_overlap: Số token overlap giữa các chunk.
        verbose: Bật log chi tiết.
        
    Returns:
        List[str]: Danh sách đoạn văn đã chunk.
    """
    if verbose:
        logger.info(f"Splitting text length={len(text)} into chunks (target ~{chunk_size} tokens each)")

    # Load tokenizer tương thích với E5
    tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base")

    headers = [("##", "Header 2"), ("###", "Header 3"), ("#", "Header 1")]

    # 1. Tách theo heading trước (cấu trúc logic)
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers, 
        # strip_headers=False
    )
    docs = markdown_splitter.split_text(text)

    # 2. Dùng TokenTextSplitter để chunk mỗi đoạn heading thành nhiều đoạn nhỏ
    token_splitter = TokenTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # 3. Chunk từng đoạn từ heading splitter
    all_chunks = []
    for doc in docs:
        content = doc.page_content.strip()
        if not content:
            continue
        chunks = token_splitter.split_text(content)
        all_chunks.extend([chunk.strip() for chunk in chunks if chunk.strip()])

    if verbose:
        logger.info(f"Split text into {len(all_chunks)} chunks (avg length: {sum(len(c) for c in all_chunks) // len(all_chunks)} chars)")

    return all_chunks
