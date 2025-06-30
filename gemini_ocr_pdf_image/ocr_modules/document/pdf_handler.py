"""
PDF document handling service for OCR processing.

This module handles PDF-specific operations including page rendering,
progress tracking, and content extraction coordination.
"""

import fitz
from pathlib import Path
from typing import Optional, Generator, Tuple
from PIL import Image
import io

from ..models import PageProgress


class PDFHandler:
    """
    Specialized handler for PDF document processing operations.
    
    Handles PDF opening, page rendering, progress tracking, and
    provides a clean interface for PDF-specific OCR operations.
    """
    
    def __init__(self):
        """Initialize the PDF handler."""
        self.document = None
        self.total_pages = 0
    
    def open_document(self, pdf_path: str) -> int:
        """
        Open a PDF document for processing.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Total number of pages in the document
            
        Raises:
            Exception: If PDF cannot be opened
        """
        try:
            self.document = fitz.open(pdf_path)
            self.total_pages = len(self.document)
            return self.total_pages
        except Exception as e:
            raise Exception(f"Failed to open PDF {pdf_path}: {str(e)}")
    
    def close_document(self):
        """Close the PDF document and free resources."""
        if self.document:
            self.document.close()
            self.document = None
            self.total_pages = 0
    
    def validate_page_range(self, start_page: int, end_page: Optional[int] = None) -> Tuple[int, int]:
        """
        Validate and adjust page range for processing.
        
        Args:
            start_page: First page to process (1-based)
            end_page: Last page to process (inclusive, 1-based)
            
        Returns:
            Tuple of (validated_start_page, validated_end_page)
        """
        if not self.document:
            raise Exception("No document is currently open")
        
        # Adjust page range
        if start_page < 1:
            start_page = 1
        if end_page is None or end_page > self.total_pages:
            end_page = self.total_pages
            
        return start_page, end_page
    
    def get_page_range(self, start_page: int, end_page: int) -> range:
        """
        Get page range for iteration (0-based indexing for fitz).
        
        Args:
            start_page: First page (1-based)
            end_page: Last page (1-based, inclusive)
            
        Returns:
            Range object for page iteration
        """
        return range(start_page - 1, end_page)  # Convert to 0-based index
    
    def render_page_as_image(self, page_index: int, dpi: int = 300) -> Image.Image:
        """
        Render a PDF page as a PIL Image.
        
        Args:
            page_index: Page index (0-based)
            dpi: Resolution for rendering
            
        Returns:
            PIL Image of the rendered page
            
        Raises:
            Exception: If page cannot be rendered
        """
        if not self.document:
            raise Exception("No document is currently open")
        
        if page_index >= self.total_pages:
            raise Exception(f"Page index {page_index} exceeds document length {self.total_pages}")
        
        try:
            # Render page at specified DPI
            page = self.document[page_index]
            mat = fitz.Matrix(dpi/72, dpi/72)  # Scale factor for DPI
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            img_data = pix.tobytes("ppm")
            image = Image.open(io.BytesIO(img_data))
            
            return image
            
        except Exception as e:
            raise Exception(f"Failed to render page {page_index + 1}: {str(e)}")
    
    def get_page_iterator(self, start_page: int, end_page: int) -> Generator[Tuple[int, int], None, None]:
        """
        Get an iterator for processing pages.
        
        Args:
            start_page: First page (1-based)
            end_page: Last page (1-based, inclusive)
            
        Yields:
            Tuple of (page_index_0_based, page_number_1_based)
        """
        page_range = self.get_page_range(start_page, end_page)
        for page_index in page_range:
            page_number = page_index + 1  # Convert to 1-based
            yield page_index, page_number
    
    def calculate_progress_summary(self, start_page: int, end_page: int, progress: dict) -> dict:
        """
        Calculate progress summary for the given page range.
        
        Args:
            start_page: First page (1-based)
            end_page: Last page (1-based, inclusive)
            progress: Progress dictionary from progress manager
            
        Returns:
            Dictionary with progress statistics
        """
        pages_to_process = list(range(start_page, end_page + 1))  # 1-based page numbers
        num_pages = len(pages_to_process)
        
        completed_pages = sum(1 for page_num in pages_to_process 
                            if page_num in progress and progress[page_num].status == 'completed')
        remaining_pages = num_pages - completed_pages
        completion_percentage = (completed_pages / num_pages) * 100 if num_pages > 0 else 0
        
        return {
            'total_pages': num_pages,
            'completed_pages': completed_pages,
            'remaining_pages': remaining_pages,
            'completion_percentage': completion_percentage,
            'pages_to_process': pages_to_process
        }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure document is closed."""
        self.close_document()