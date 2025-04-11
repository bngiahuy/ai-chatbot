import os
import logging

logger = logging.getLogger(__name__)

def get_pdf_files(directory):
    """
    Get all PDF files in the specified directory.
    
    Args:
        directory (str): Directory to search for PDF files.
        
    Returns:
        list: List of PDF file paths.
    """
    logger.info(f"Scanning for PDF files in: {directory}")
    pdf_files = []
    
    for file in os.listdir(directory):
        if file.lower().endswith('.pdf'):
            pdf_files.append(os.path.join(directory, file))
    
    logger.info(f"Found {len(pdf_files)} PDF files")
    return pdf_files

def get_md_files(directory):
    """
    Get all Markdown files in the specified directory.
    
    Args:
        directory (str): Directory to search for Markdown files.
        
    Returns:
        list: List of Markdown file paths.
    """
    logger.info(f"Scanning for Markdown files in: {directory}")
    md_files = []
    
    for file in os.listdir(directory):
        if file.lower().endswith('.md'):
            md_files.append(os.path.join(directory, file))
    
    logger.info(f"Found {len(md_files)} Markdown files")
    return md_files