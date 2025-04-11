import os
import logging
import re
import unicodedata
from pdfminer.high_level import extract_text

logger = logging.getLogger(__name__)

def extract_text_from_markdown(markdown_path):
    """
    Extract text from a Markdown file while preserving its structure.
    
    Args:
        markdown_path (str): Path to the Markdown file.
    
    Returns:
        str: Processed text from the Markdown file with structure preserved.
    """
    if not os.path.exists(markdown_path):
        logger.error(f"File not found: {markdown_path}")
        return ""
    
    logger.info(f"Extracting text from markdown file: {markdown_path}")
    
    try:
        with open(markdown_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Số ký tự ban đầu
        original_length = len(content)
        logger.info(f"Read {original_length} characters from {os.path.basename(markdown_path)}")
        
        # Xử lý cơ bản để chuẩn hóa nội dung markdown
        processed_content = process_markdown_content(content)
        
        logger.info(f"Processed markdown content: {len(processed_content)} characters")
        return processed_content
        
    except Exception as e:
        logger.error(f"Error extracting text from {markdown_path}: {str(e)}")
        return ""

def process_markdown_content(content):
    """
    Process markdown content to normalize it and preserve structure.
    
    Args:
        content (str): Raw markdown content.
    
    Returns:
        str: Processed markdown content.
    """
    # Chuẩn hóa Unicode
    content = unicodedata.normalize("NFC", content)
    
    # Xử lý code blocks - thay thế bằng placeholder để tránh xử lý bên trong
    code_blocks = []
    def replace_code_block(match):
        code_blocks.append(match.group(0))
        return f"[CODE_BLOCK_{len(code_blocks)-1}]"
    
    content = re.sub(r'```[\s\S]*?```', replace_code_block, content)
    
    # Đảm bảo tiêu đề markdown có khoảng trống phía trước
    content = re.sub(r'(?<!\n)\n#', '\n\n#', content)
    
    # Đảm bảo danh sách có định dạng nhất quán
    content = re.sub(r'(?<!\n)\n([*\-+]|\d+\.)\s', r'\n\n\1 ', content)
    
    # Xử lý liên kết markdown - giữ lại văn bản hiển thị
    content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', content)
    
    # Xử lý hình ảnh markdown - giữ lại alt text
    content = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'Image: \1', content)
    
    # Khôi phục code blocks
    for i, block in enumerate(code_blocks):
        content = content.replace(f"[CODE_BLOCK_{i}]", block)
    
    # Đảm bảo khoảng trống nhất quán giữa các đoạn văn
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    return content.strip()

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file using pdfminer.six.
    
    Args:
        pdf_path (str): Path to the PDF file.
        
    Returns:
        str: Extracted text from the PDF.
    """
    logger.info(f"Extracting text from {pdf_path}")
    try:
        text = extract_text(pdf_path)
        logger.info(f"Extracted {len(text)} characters from {os.path.basename(pdf_path)}")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
        return ""