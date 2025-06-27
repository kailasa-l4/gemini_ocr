"""
OCR Modules Package

This package contains the modularized components of the Enhanced OCR processor.
"""

from .ocr_processor_modular import GeminiAdvancedOCR
from .models import (
    LegibilityResult,
    CombinedAssessmentResult, 
    OCRResult,
    SemanticResult,
    PageProgress,
    ImageProgress
)
from .utils import is_pdf_file, is_image_file, find_image_files, find_all_supported_files, get_safe_filename

__all__ = [
    'GeminiAdvancedOCR',
    'LegibilityResult',
    'CombinedAssessmentResult',
    'OCRResult', 
    'SemanticResult',
    'PageProgress',
    'ImageProgress',
    'is_pdf_file',
    'is_image_file',
    'find_image_files',
    'find_all_supported_files',
    'get_safe_filename'
]