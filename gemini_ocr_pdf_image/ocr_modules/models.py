"""
Data models for the OCR processor.
"""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class LegibilityResult:
    is_legible: bool
    confidence_score: float
    text_clarity: str
    image_quality: str
    reason: str
    processing_time: float


@dataclass
class CombinedAssessmentResult:
    """Combined legibility and semantic pre-assessment result."""
    should_process: bool
    legibility_score: float
    expected_semantic_quality: float
    text_clarity: str
    image_quality: str
    visible_text_sample: str
    language_detected: str
    ocr_prediction: str
    semantic_prediction: str
    issues_found: List[str]
    reason: str
    processing_time: float


@dataclass
class OCRResult:
    extracted_text: str
    confidence: float
    language_detected: str
    processing_time: float


@dataclass
class SemanticResult:
    is_meaningful: bool
    semantic_score: float
    language_consistency: str
    word_formation: str
    coherence_level: str
    issues_found: List[str]
    reason: str
    processing_time: float


@dataclass
class PageProgress:
    page_num: int
    status: str  # 'pending', 'illegible', 'semantically_invalid', 'completed', 'error'
    legibility_score: Optional[float]
    semantic_score: Optional[float]
    ocr_confidence: Optional[float]
    processing_time: float
    error_message: Optional[str]
    timestamp: str
    # Assessment details
    text_clarity: Optional[str] = None
    image_quality: Optional[str] = None
    ocr_prediction: Optional[str] = None
    semantic_prediction: Optional[str] = None
    visible_text_sample: Optional[str] = None
    language_detected: Optional[str] = None
    issues_found: Optional[str] = None  # Comma-separated string


@dataclass
class ImageProgress:
    file_path: str
    status: str  # 'pending', 'illegible', 'semantically_invalid', 'completed', 'error'
    legibility_score: Optional[float]
    semantic_score: Optional[float]
    ocr_confidence: Optional[float]
    processing_time: float
    error_message: Optional[str]
    timestamp: str
    # Assessment details
    text_clarity: Optional[str] = None
    image_quality: Optional[str] = None
    ocr_prediction: Optional[str] = None
    semantic_prediction: Optional[str] = None
    visible_text_sample: Optional[str] = None
    language_detected: Optional[str] = None
    issues_found: Optional[str] = None  # Comma-separated string