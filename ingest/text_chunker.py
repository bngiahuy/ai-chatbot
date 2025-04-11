from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
import re
from typing import List

logger = logging.getLogger(__name__)

def split_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    preserve_bullets: bool = True,
    verbose: bool = False
) -> List[str]:
    """
    Tách văn bản markdown thành các đoạn (chunk) nhỏ theo đoạn văn, 
    có hỗ trợ overlap và bảo toàn block bullet list.

    Args:
        text (str): Văn bản đầu vào (markdown).
        chunk_size (int): Số ký tự tối đa mỗi chunk.
        chunk_overlap (int): Số ký tự trùng giữa các chunk.
        preserve_bullets (bool): Gộp các bullet list liên tục thành một block.
        verbose (bool): Bật chế độ debug/log.

    Returns:
        List[str]: Danh sách các chunk đã chia.
    """

    # Normalize endlines
    lines = text.strip().splitlines()

    # Gom đoạn văn dựa trên dòng trắng
    paragraphs = []
    buffer = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if buffer:
                paragraphs.append("\n".join(buffer).strip())
                buffer = []
        elif preserve_bullets and re.match(r"^[-*+]\s", stripped):
            # Bullet item: gộp với dòng trước nếu là bullet
            buffer.append(stripped)
        else:
            # Header, đoạn văn, v.v.
            if preserve_bullets and buffer and all(re.match(r"^[-*+]\s", l.strip()) for l in buffer):
                paragraphs.append("\n".join(buffer).strip())
                buffer = [stripped]
            else:
                buffer.append(stripped)

    if buffer:
        paragraphs.append("\n".join(buffer).strip())

    if verbose:
        logger.info(f"[INFO] Tổng số đoạn văn: {len(paragraphs)}")

    # Gộp các đoạn vào chunk theo chunk_size và chunk_overlap
    chunks = []
    current_chunk = ""
    i = 0
    while i < len(paragraphs):
        para = paragraphs[i]

        # Nếu thêm đoạn này không vượt quá chunk_size → thêm vào
        if len(current_chunk) + len(para) + 1 <= chunk_size:
            current_chunk += para + "\n\n"
            i += 1
        else:
            # Nếu chunk quá ngắn thì vẫn thêm đoạn mới vào cho đủ
            if not current_chunk:
                current_chunk = para[:chunk_size]
                i += 1

            trimmed_chunk = current_chunk.strip()
            chunks.append(trimmed_chunk)

            # Tạo overlap từ cuối chunk hiện tại
            if chunk_overlap > 0 and len(trimmed_chunk) > chunk_overlap:
                overlap_text = trimmed_chunk[-chunk_overlap:]
                current_chunk = overlap_text
            else:
                current_chunk = ""

    # Thêm chunk cuối nếu còn nội dung
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    if verbose:
        logger.info(f"[INFO] Tổng số chunk tạo ra: {len(chunks)}")
        total_input = len(text)
        total_output = sum(len(c) for c in chunks)
        logger.info(f"[INFO] Tổng số ký tự đầu vào: {total_input}")
        logger.info(f"[INFO] Tổng số ký tự đầu ra: {total_output}")
        for idx, ch in enumerate(chunks):
            logger.info(f"[DEBUG] Chunk {idx+1}: {len(ch)} ký tự, preview:\n{ch[:100]}...\n")

    return chunks