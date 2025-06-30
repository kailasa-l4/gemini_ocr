"""
Modular OCR Processor - Main class that integrates all modules.

This is the refactored version of GeminiAdvancedOCR that uses the modular components.
"""

from .ocr_engine import GeminiOCREngine
from .processors import PDFProcessor, ImageProcessor, ImageDirectoryProcessor
from .utils import is_pdf_file, is_image_file


class GeminiAdvancedOCR:
    """Main OCR processor class that orchestrates all components."""
    
    def __init__(self, api_key: str, thinking_budget: int = 2000, 
                 enable_thinking_assessment: bool = True, enable_thinking_ocr: bool = False,
                 db_logger=None, logs_dir: str = './logs', verbose: bool = False):
        """Initialize the modular OCR processor."""
        # Store database logger and logs directory
        self.db_logger = db_logger
        self.logs_dir = logs_dir
        
        # Store model configurations
        self.assessment_model = 'gemini-2.5-flash'  # Default
        self.ocr_model = 'gemini-2.5-flash'  # Default
        
        # Initialize the OCR engine
        self.ocr_engine = GeminiOCREngine(
            api_key, thinking_budget, enable_thinking_assessment, enable_thinking_ocr,
            db_logger=db_logger, logs_dir=logs_dir, verbose=verbose,
            assessment_model=self.assessment_model, ocr_model=self.ocr_model
        )
        
        # Initialize processors
        self.pdf_processor = PDFProcessor(self.ocr_engine, db_logger=db_logger, logs_dir=logs_dir)
        self.image_processor = ImageProcessor(self.ocr_engine, db_logger=db_logger, logs_dir=logs_dir)
        self.directory_processor = ImageDirectoryProcessor(self.ocr_engine, db_logger=db_logger, logs_dir=logs_dir)
    
    # Delegate methods to the OCR engine for backward compatibility
    def combined_pre_assessment(self, image, page_num, legibility_threshold=0.5, semantic_threshold=0.6):
        """Combined legibility and semantic pre-assessment."""
        return self.ocr_engine.combined_pre_assessment(image, page_num, legibility_threshold, semantic_threshold)
    
    def assess_legibility(self, image, page_num):
        """Legacy method for legibility assessment."""
        return self.ocr_engine.assess_legibility(image, page_num)
    
    def validate_semantic_meaning(self, text, language, page_num):
        """Legacy method for semantic validation."""
        return self.ocr_engine.validate_semantic_meaning(text, language, page_num)
    
    def extract_text(self, image, page_num):
        """Extract text from image."""
        return self.ocr_engine.extract_text(image, page_num)
    
    
    # Delegate methods to processors
    def process_pdf(self, pdf_path, output_dir, start_page=1, end_page=None, dpi=300, 
                    legibility_threshold=0.5, semantic_threshold=0.6, file_progress=None):
        """Process a PDF file."""
        return self.pdf_processor.process_pdf(
            pdf_path, output_dir, start_page, end_page, dpi, 
            legibility_threshold, semantic_threshold, file_progress=file_progress
        )
    
    def process_single_image(self, image_path, output_dir, legibility_threshold=0.5, 
                           semantic_threshold=0.6, skip_text_cleaning=False):
        """Process a single image file."""
        return self.image_processor.process_single_image(
            image_path, output_dir, legibility_threshold, semantic_threshold, skip_text_cleaning
        )
    
    def process_images(self, input_dir, output_dir, legibility_threshold=0.5, 
                      semantic_threshold=0.6, skip_text_cleaning=False):
        """Process a directory of images."""
        return self.directory_processor.process_images(
            input_dir, output_dir, legibility_threshold, semantic_threshold, skip_text_cleaning
        )
    
    # Utility methods
    def is_pdf_file(self, file_path):
        """Check if file is a PDF."""
        return is_pdf_file(file_path)
    
    def is_image_file(self, file_path):
        """Check if file is a supported image format."""
        return is_image_file(file_path)
    
    def is_pdf_fully_completed(self, pdf_path, output_dir, start_page=1, end_page=None):
        """
        Fast check if a PDF file is already fully processed.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Output directory
            start_page: First page to check (1-based)
            end_page: Last page to check (inclusive)
            
        Returns:
            True if all pages are completed, False otherwise
        """
        import fitz
        from pathlib import Path
        
        try:
            # Get basic info about the PDF
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            doc.close()
            
            # Adjust page range like in the processor
            if start_page < 1:
                start_page = 1
            if end_page is None or end_page > total_pages:
                end_page = total_pages
                
            num_pages = end_page - start_page + 1
            
            # Setup paths like in the processor
            book_name = Path(pdf_path).stem
            book_output_dir = Path(output_dir) / book_name
            progress_file = book_output_dir / f"{book_name}_progress.csv"
            
            # Fast completion check using the optimized method
            completion_status = self.pdf_processor.progress_manager.get_completion_status_fast(
                str(progress_file), num_pages
            )
            
            # Return True only if ALL pages are completed
            return completion_status['remaining_count'] == 0
            
        except Exception as e:
            # If any error occurs, assume not completed (safe default)
            print(f"Warning: Could not check completion status for {pdf_path}: {e}")
            return False
    
    # Legacy progress methods (delegated to processors)
    def load_progress(self, progress_file):
        """Load page progress from CSV file."""
        return self.pdf_processor.progress_manager.load_page_progress(progress_file)
    
    def save_progress(self, progress, progress_file):
        """Save page progress to CSV file."""
        return self.pdf_processor.progress_manager.save_page_progress(progress, progress_file)
    
    def load_image_progress(self, progress_file):
        """Load image progress from CSV file."""
        return self.directory_processor.progress_manager.load_image_progress(progress_file)
    
    def save_image_progress(self, progress, progress_file):
        """Save image progress to CSV file."""
        return self.directory_processor.progress_manager.save_image_progress(progress, progress_file)
    
    def get_safe_filename(self, relative_path):
        """Convert relative path to safe filename."""
        from .utils import get_safe_filename
        return get_safe_filename(relative_path)
    
    def find_image_files(self, input_dir):
        """Find all image files in directory."""
        from .utils import find_image_files
        return find_image_files(input_dir)