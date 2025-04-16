import os
import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

def extract_text_from_markdown(markdown_path):
    """
    Trích xuất và xử lý nội dung Markdown tiếng Việt, đảm bảo giữ nguyên cấu trúc ban đầu.
    
    Args:
        markdown_path (str): Đường dẫn tới file Markdown.
    
    Returns:
        str: Văn bản đã xử lý từ file Markdown.
    """
    if not os.path.exists(markdown_path):
        logger.error(f"Không tìm thấy file: {markdown_path}")
        return ""
    
    logger.info(f"Đang trích xuất nội dung từ file markdown: {markdown_path}")
    try:
        with open(markdown_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        original_length = len(content)
        logger.info(f"Đọc được {original_length} ký tự từ {os.path.basename(markdown_path)}")
        
        processed_content = process_markdown_content(content)
        logger.info(f"Nội dung sau xử lý có độ dài: {len(processed_content)} ký tự")
        return processed_content
    except Exception as e:
        logger.error(f"Lỗi khi xử lý file {markdown_path}: {e}")
        return ""

def process_markdown_content(content):
    """
    Xử lý nội dung Markdown tiếng Việt:
      - Chuẩn hóa Unicode.
      - Thay thế code block và inline code bằng placeholder để bảo toàn nội dung.
      - Chuẩn hóa tiêu đề và danh sách.
      - Xử lý liên kết và hình ảnh.
      - Gộp các khoảng cách dòng dư thừa.
    
    Args:
        content (str): Nội dung Markdown gốc.
    
    Returns:
        str: Nội dung đã được làm sạch và định dạng lại.
    """
    # 1. Chuẩn hóa Unicode (NFC để xử lý tiếng Việt đúng cách)
    content = unicodedata.normalize("NFC", content)
    
    # # 2. Xử lý code block: thay thế bằng placeholder để tránh xử lý nội dung bên trong
    # code_blocks = []
    # def code_block_placeholder(match):
    #     code_blocks.append(match.group(0))
    #     return f"[CODE_BLOCK_{len(code_blocks)-1}]"
    
    # content = re.sub(r'```[\s\S]*?```', code_block_placeholder, content)
    
    # # 3. Xử lý inline code: dấu `` `code` ``
    # inline_code_blocks = []
    # def inline_code_placeholder(match):
    #     inline_code_blocks.append(match.group(0))
    #     return f"[INLINE_CODE_{len(inline_code_blocks)-1}]"
    
    # content = re.sub(r'`([^`]+)`', inline_code_placeholder, content)
    
    # # 4. Chuẩn hóa định dạng tiêu đề: đảm bảo có khoảng trống trước mỗi header
    # content = re.sub(r'(?<!\n)\n(#+)', r'\n\n\1', content)
    
    # # 5. Chuẩn hóa danh sách: đảm bảo các list (bullet hoặc numbered) có dòng trống trước
    # content = re.sub(r'(?<!\n)\n([*\-+]|\d+\.)\s', r'\n\n\1 ', content)
    
    # # 6. Xử lý liên kết Markdown: giữ lại nội dung hiển thị và loại bỏ URL
    # content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', content)
    
    # # 7. Xử lý hình ảnh Markdown: giữ lại alt text (nếu có) hoặc ghi chú "Image"
    # content = re.sub(r'!\[([^\]]*)\]\([^)]+\)', lambda m: f"Image: {m.group(1).strip()}" if m.group(1).strip() else "Image", content)
    
    # # 8. Khôi phục inline code và code blocks để không mất nội dung ban đầu
    # for idx, block in enumerate(inline_code_blocks):
    #     content = content.replace(f"[INLINE_CODE_{idx}]", block)
        
    # for idx, block in enumerate(code_blocks):
    #     content = content.replace(f"[CODE_BLOCK_{idx}]", block)
    
    # # 9. Gộp khoảng cách dòng: chỉ cho phép tối đa 2 dòng trống liên tiếp
    # content = re.sub(r'\n{3,}', '\n\n', content)
    
    return content.strip()
