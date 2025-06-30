"""
Utility functions for the OCR processor.
"""

import os
import re
from pathlib import Path
from typing import List, Tuple


def is_pdf_file(file_path: str) -> bool:
    """Check if the file is a PDF."""
    return Path(file_path).suffix.lower() == '.pdf'


def is_image_file(file_path: str) -> bool:
    """Check if the file is a supported image format."""
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    return Path(file_path).suffix.lower() in supported_formats


def find_image_files(input_dir: str) -> List[Tuple[str, str]]:
    """
    Find all image files in a directory and subdirectories.
    
    Returns:
        List of tuples (file_path, relative_path)
    """
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    image_files = []
    
    input_path = Path(input_dir)
    
    for file_path in input_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in supported_formats:
            # Get relative path for organization
            relative_path = file_path.relative_to(input_path)
            image_files.append((str(file_path), str(relative_path)))
    
    # Sort by relative path for consistent processing order
    image_files.sort(key=lambda x: x[1])
    return image_files


def find_pdf_files(input_dir: str) -> List[str]:
    """
    Find all PDF files in a directory and subdirectories.
    
    Returns:
        List of PDF file paths
    """
    pdf_files = []
    input_path = Path(input_dir)
    
    for file_path in input_path.rglob('*.pdf'):
        if file_path.is_file():
            pdf_files.append(str(file_path))
    
    # Sort for consistent processing order
    pdf_files.sort()
    return pdf_files


def find_all_supported_files(input_dir: str) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Find all supported files (PDFs and images) in a directory.
    
    Returns:
        Tuple of (pdf_files, image_files)
        - pdf_files: List of PDF file paths
        - image_files: List of tuples (file_path, relative_path)
    """
    pdf_files = find_pdf_files(input_dir)
    image_files = find_image_files(input_dir)
    return pdf_files, image_files


def get_safe_filename(relative_path: str) -> str:
    """Convert relative path to safe filename for markdown files."""
    # Replace path separators and special characters
    safe_name = str(relative_path).replace('/', '_').replace('\\', '_')
    safe_name = re.sub(r'[<>:"|?*]', '_', safe_name)
    # Remove file extension and add index if too long
    safe_name = Path(safe_name).stem
    if len(safe_name) > 100:
        safe_name = safe_name[:100]
    return safe_name