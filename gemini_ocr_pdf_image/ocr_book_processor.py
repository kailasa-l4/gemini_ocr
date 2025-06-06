"""
Enhanced OCR Script with Gemini 2.5 Flash and Combined Pre-Assessment

Features:
- Supports PDF files, single image files, and directories of image files
- Optimized combined pre-assessment: Visual legibility + Semantic quality prediction in one API call
- Structured outputs using JSON schemas
- Progress tracking with CSV files
- Resume functionality for interrupted processing
- Markdown output for all extracted text
- Recursive directory scanning for images
- Configurable thresholds for both visual and semantic validation
- Reduced API consumption (2 calls per image instead of 3)

Supported image formats: JPG, JPEG, PNG, BMP, TIFF, TIF, WEBP

Usage:
    # Process a PDF file
    python ocr_book_processor.py --input-file book.pdf --output-dir ./output --api-key YOUR_KEY
    
    # Process a single image file with custom thresholds
    python ocr_book_processor.py --input-file old_document.png --output-dir ./output --api-key YOUR_KEY \
        --legibility-threshold 0.6 --semantic-threshold 0.7
    
    # Process a directory of images
    python ocr_book_processor.py --input-dir ./images --output-dir ./output --api-key YOUR_KEY
"""

import os
import re
import json
import time
import argparse
import tempfile
import csv
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import fitz  # PyMuPDF
from google import genai
from google.genai import types
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import mimetypes

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

class GeminiAdvancedOCR:
    """Enhanced OCR processor with legibility detection using Gemini 2.5 Flash."""
    
    def __init__(self, api_key: str, thinking_budget: int = 2000):
        """Initialize the OCR processor with new Google Gen AI SDK."""
        self.api_key = api_key
        self.thinking_budget = thinking_budget
        
        # Initialize the new Google Gen AI client
        self.client = genai.Client(api_key=api_key)
        
        # Model configuration with thinking enabled
        self.model_config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=None,  # Will be set per request
            max_output_tokens=4000,
            temperature=0.1
        )
        
        # Store thinking budget for per-request configuration
        self.thinking_budget = thinking_budget
        print(f"Initialized OCR processor with thinking budget: {thinking_budget}")
        
        # Legibility assessment schema
        self.legibility_schema = {
            "type": "object",
            "properties": {
                "is_legible": {
                    "type": "boolean",
                    "description": "Whether the text in the image is legible enough for OCR"
                },
                "confidence_score": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Confidence score for legibility assessment (0-1)"
                },
                "text_clarity": {
                    "type": "string",
                    "enum": ["poor", "fair", "good", "excellent"],
                    "description": "Overall clarity of text in the image"
                },
                "image_quality": {
                    "type": "string", 
                    "enum": ["poor", "fair", "good", "excellent"],
                    "description": "Overall quality of the image"
                },
                "reason": {
                    "type": "string",
                    "description": "Detailed explanation for the legibility assessment"
                }
            },
            "required": ["is_legible", "confidence_score", "text_clarity", "image_quality", "reason"]
        }
        
        # OCR extraction schema
        self.ocr_schema = {
            "type": "object",
            "properties": {
                "extracted_text": {
                    "type": "string",
                    "description": "The complete text extracted from the image"
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Confidence score for the OCR extraction"
                },
                "language_detected": {
                    "type": "string",
                    "description": "Primary language detected in the text"
                }
            },
            "required": ["extracted_text", "confidence", "language_detected"]
        }
        
        # Semantic validation schema
        self.semantic_schema = {
            "type": "object",
            "properties": {
                "is_meaningful": {
                    "type": "boolean",
                    "description": "Whether the extracted text is semantically meaningful"
                },
                "semantic_score": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Score for semantic meaningfulness (0-1)"
                },
                "language_consistency": {
                    "type": "string",
                    "enum": ["consistent", "mixed", "inconsistent", "gibberish"],
                    "description": "Consistency of language usage"
                },
                "word_formation": {
                    "type": "string",
                    "enum": ["valid", "mostly_valid", "poor", "invalid"],
                    "description": "Quality of word formation in the detected language"
                },
                "coherence_level": {
                    "type": "string",
                    "enum": ["high", "medium", "low", "none"],
                    "description": "Level of textual coherence and readability"
                },
                "issues_found": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of specific issues identified in the text"
                },
                "reason": {
                    "type": "string",
                    "description": "Detailed explanation of the semantic assessment"
                }
            },
            "required": ["is_meaningful", "semantic_score", "language_consistency", 
                        "word_formation", "coherence_level", "reason"]
        }
        
        # Combined pre-assessment schema
        self.combined_assessment_schema = {
            "type": "object",
            "properties": {
                "should_process": {
                    "type": "boolean",
                    "description": "Whether this image should proceed to full OCR processing"
                },
                "legibility_score": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Visual legibility score (0-1)"
                },
                "expected_semantic_quality": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Predicted semantic quality - will OCR produce meaningful text? (0-1)"
                },
                "text_clarity": {
                    "type": "string",
                    "enum": ["poor", "fair", "good", "excellent"],
                    "description": "Overall clarity of text in the image"
                },
                "image_quality": {
                    "type": "string",
                    "enum": ["poor", "fair", "good", "excellent"],
                    "description": "Overall quality of the image"
                },
                "visible_text_sample": {
                    "type": "string",
                    "description": "Sample of text you can actually read clearly from the image"
                },
                "language_detected": {
                    "type": "string",
                    "description": "Primary language/script detected from visible text"
                },
                "ocr_prediction": {
                    "type": "string",
                    "enum": ["excellent", "good", "fair", "poor", "unusable"],
                    "description": "Predicted OCR result quality based on what you can see"
                },
                "semantic_prediction": {
                    "type": "string",
                    "enum": ["meaningful_text", "partial_meaning", "fragmented", "mostly_gibberish", "unusable"],
                    "description": "Expected meaningfulness of OCR output"
                },
                "issues_found": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific issues that would affect OCR quality"
                },
                "reason": {
                    "type": "string",
                    "description": "Detailed explanation of your analysis and decision"
                }
            },
            "required": ["should_process", "legibility_score", "expected_semantic_quality", 
                        "text_clarity", "image_quality", "visible_text_sample", 
                        "language_detected", "ocr_prediction", "semantic_prediction", "issues_found", "reason"]
        }
    
    def combined_pre_assessment(self, image: Image.Image, page_num: int, 
                              legibility_threshold: float = 0.5, 
                              semantic_threshold: float = 0.6) -> CombinedAssessmentResult:
        """Combined legibility and semantic pre-assessment to determine if OCR should be attempted."""
        start_time = time.time()
        
        prompt = f"""
        THINK CAREFULLY about this scanned document image (Page {page_num}) to determine if OCR processing will produce meaningful, usable text.
        
        **CRITICAL THINKING PROCESS:**
        
        STEP 1 - VISUAL ANALYSIS:
        Look at the image and think through:
        - Can I clearly distinguish individual letters and characters?
        - Is the text sharp and well-defined, or blurry and unclear?
        - Are there obvious damage, stains, or degradation issues?
        - Is the lighting/contrast sufficient to read the text?
        - What's the overall image quality and resolution?
        
        STEP 2 - TEXT LEGIBILITY EXAMINATION:
        Examine the visible text carefully:
        - Can I actually READ specific words or characters in this image?
        - Are the letters clear enough that I could transcribe them accurately?
        - Do I see complete, well-formed characters or just unclear marks?
        - Are there obvious OCR challenges (handwriting, unusual fonts, damage)?
        
        STEP 3 - SEMANTIC PREDICTION:
        This is CRITICAL - think about what OCR results you would expect:
        - Based on what I can see, would OCR produce complete, meaningful words?
        - Do the visible characters form recognizable words in a known language?
        - Or would OCR likely produce fragments, gibberish, or meaningless character combinations?
        - Are there signs this would result in nonsensical output (damaged text, unclear script, poor quality)?
        
        STEP 4 - LANGUAGE AND SCRIPT ANALYSIS:
        - What language/script is this? (Tamil, Sanskrit, English, Hindi, etc.)
        - Is the script clear enough for accurate character recognition?
        - Are there diacritical marks or special characters that need to be preserved?
        - Does this script typically OCR well, or are there known challenges?
        
        STEP 5 - QUALITY PREDICTION:
        Based on your analysis, predict:
        - Legibility score: How clearly can the text be seen? ({legibility_threshold} minimum needed)
        - Semantic quality: Would OCR produce meaningful, usable text? ({semantic_threshold} minimum needed)
        
        **DECISION CRITERIA - BE STRICT:**
        Set should_process=true ONLY if you are confident that:
        1. The text is visually clear enough for accurate OCR (legibility >= {legibility_threshold})
        2. OCR would produce meaningful, coherent text rather than gibberish (semantic >= {semantic_threshold})
        
        **IMPORTANT:** If you see signs that OCR would produce fragmented words, nonsensical combinations, or mostly gibberish, set a LOW semantic quality score (< {semantic_threshold}) to prevent wasting processing time.
        
        Provide a sample of the most clearly visible text you can actually read from the image.
        """
        
        # Configure for combined assessment with thinking
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=self.combined_assessment_schema,
            max_output_tokens=4000,
            temperature=0.1
        )
        
        # Add thinking configuration for this specific request
        try:
            config.thinking_config = types.ThinkingConfig(
                thinking_budget=self.thinking_budget
            )
            print(f"Applied thinking config with budget: {self.thinking_budget}")
        except Exception as e:
            print(f"Could not apply thinking config: {str(e)}")
        
        # Convert PIL image to bytes for the API
        import io
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash-preview-05-20',
                contents=[
                    prompt,
                    types.Part.from_bytes(
                        data=img_bytes.getvalue(), 
                        mime_type='image/png'
                    )
                ],
                config=config
            )
            
            # Check if response is valid
            if response is None or not hasattr(response, 'text') or response.text is None:
                raise Exception("API returned None response or missing text content")
            
            # Parse JSON with better error handling
            try:
                result_data = json.loads(response.text)
            except json.JSONDecodeError as je:
                print(f"JSON parsing error in combined assessment for page {page_num}: {str(je)}")
                print(f"Problematic response text: {response.text[:500]}...")
                raise Exception(f"JSON parsing failed: {str(je)}")
            processing_time = time.time() - start_time
            
            return CombinedAssessmentResult(
                should_process=result_data['should_process'],
                legibility_score=result_data['legibility_score'],
                expected_semantic_quality=result_data['expected_semantic_quality'],
                text_clarity=result_data['text_clarity'],
                image_quality=result_data['image_quality'],
                visible_text_sample=result_data['visible_text_sample'],
                language_detected=result_data['language_detected'],
                ocr_prediction=result_data['ocr_prediction'],
                semantic_prediction=result_data['semantic_prediction'],
                issues_found=result_data['issues_found'],
                reason=result_data['reason'],
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"Error in combined assessment for page {page_num}: {str(e)}")
            return CombinedAssessmentResult(
                should_process=False,
                legibility_score=0.0,
                expected_semantic_quality=0.0,
                text_clarity="poor",
                image_quality="poor",
                visible_text_sample="",
                language_detected="unknown",
                ocr_prediction="unusable",
                semantic_prediction="unusable",
                issues_found=[f"Assessment error: {str(e)}"],
                reason=f"Error during combined assessment: {str(e)}",
                processing_time=processing_time
            )
    
    def assess_legibility(self, image: Image.Image, page_num: int) -> LegibilityResult:
        """Assess if the image text is legible enough for OCR processing."""
        start_time = time.time()
        
        prompt = f"""
        Analyze this scanned page image (Page {page_num}) to determine if the text is legible enough for OCR processing.
        
        Assessment criteria:
        1. Text clarity: Can you clearly see individual characters and words?
        2. Image quality: Is the image resolution and contrast sufficient?
        3. Legibility threshold: Would a human be able to read this text without significant difficulty?
        
        Consider factors like:
        - Blur, pixelation, or distortion
        - Poor contrast or lighting
        - Damaged or faded text
        - Extremely small or large text
        - Handwriting vs printed text quality
        - Background noise or interference
        
        Return a confidence score where:
        - 0.8-1.0: Excellent legibility, OCR will work very well
        - 0.6-0.8: Good legibility, OCR should work well
        - 0.4-0.6: Fair legibility, OCR may have issues
        - 0.2-0.4: Poor legibility, OCR likely to fail
        - 0.0-0.2: Very poor, don't attempt OCR
        
        Set is_legible=true only if confidence_score >= 0.5
        """
        
        # Configure for legibility assessment
        config = self.model_config
        config.response_schema = self.legibility_schema
        
        # Convert PIL image to bytes for the API
        import io
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash-preview-05-20',
                contents=[
                    prompt,
                    types.Part.from_bytes(
                        data=img_bytes.getvalue(), 
                        mime_type='image/png'
                    )
                ],
                config=config
            )
            
            result_data = json.loads(response.text)
            processing_time = time.time() - start_time
            
            return LegibilityResult(
                is_legible=result_data['is_legible'],
                confidence_score=result_data['confidence_score'],
                text_clarity=result_data['text_clarity'],
                image_quality=result_data['image_quality'],
                reason=result_data['reason'],
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"Error in legibility assessment for page {page_num}: {str(e)}")
            return LegibilityResult(
                is_legible=False,
                confidence_score=0.0,
                text_clarity="poor",
                image_quality="poor",
                reason=f"Error during assessment: {str(e)}",
                processing_time=processing_time
            )
    
    def validate_semantic_meaning(self, text: str, language: str, page_num: int) -> SemanticResult:
        """
        Validate if the extracted text is semantically meaningful.
        
        Args:
            text: Extracted text to validate
            language: Detected language of the text
            page_num: Page number for logging
            
        Returns:
            SemanticResult with validation details
        """
        start_time = time.time()
        
        prompt = f"""
        Analyze this extracted OCR text to determine if it's semantically meaningful and coherent in {language}.
        
        Text to analyze:
        ```
        {text}
        ```
        
        Evaluation criteria:
        
        1. **Language Consistency**: 
           - Is the text consistently in {language}?
           - Are there random character combinations that don't form words?
           - Is there mixing of scripts or languages inappropriately?
        
        2. **Word Formation**: 
           - Do the words follow proper formation rules for {language}?
           - Are there valid morphological structures?
           - Do you recognize actual words in the language?
        
        3. **Coherence Level**:
           - Do sentences make logical sense?
           - Is there proper sentence structure?
           - Does the text flow naturally?
        
        4. **Common OCR Issues to Check**:
           - Repeated gibberish patterns
           - Random character combinations
           - Impossible letter sequences
           - Missing spaces or word boundaries
           - Character substitution errors creating nonsense
        
        5. **Context Considerations**:
           - This might be from an old document with archaic language
           - Script might be historical or regional variant
           - Document might be damaged affecting readability
        
        **Scoring Guidelines:**
        - 0.9-1.0: Clearly meaningful, well-formed text
        - 0.7-0.9: Mostly meaningful with minor issues
        - 0.5-0.7: Partially meaningful, some words recognizable
        - 0.3-0.5: Poor quality, few recognizable elements
        - 0.0-0.3: Mostly gibberish, not useful for processing
        
        Set is_meaningful=true only if the text appears to contain recognizable words and coherent meaning.
        
        Provide specific issues found and detailed reasoning for your assessment.
        """
        
        # Configure for semantic validation
        config = self.model_config
        config.response_schema = self.semantic_schema
        
        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash-preview-05-20',
                contents=[prompt],
                config=config
            )
            
            result_data = json.loads(response.text)
            processing_time = time.time() - start_time
            
            return SemanticResult(
                is_meaningful=result_data['is_meaningful'],
                semantic_score=result_data['semantic_score'],
                language_consistency=result_data['language_consistency'],
                word_formation=result_data['word_formation'],
                coherence_level=result_data['coherence_level'],
                issues_found=result_data.get('issues_found', []),
                reason=result_data['reason'],
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"Error in semantic validation for page {page_num}: {str(e)}")
            return SemanticResult(
                is_meaningful=False,
                semantic_score=0.0,
                language_consistency="inconsistent",
                word_formation="invalid",
                coherence_level="none",
                issues_found=[f"Validation error: {str(e)}"],
                reason=f"Error during semantic validation: {str(e)}",
                processing_time=processing_time
            )
    
    def extract_text(self, image: Image.Image, page_num: int) -> OCRResult:
        """Extract text from a legible image using OCR."""
        start_time = time.time()
        
        prompt = f"""
        Extract ALL text from this scanned page image (Page {page_num}). This document may contain text in various languages including English, Sanskrit, Tamil, Hindi, or other Indian languages.
        
        Guidelines:
        1. Capture ALL visible text including headings, paragraphs, footnotes, captions
        2. Preserve the original text structure and formatting as much as possible
        3. Maintain paragraph breaks and section organization
        4. For Sanskrit text: Preserve all diacritical marks (ā, ī, ū, ṛ, ṝ, ḷ, ḹ, ṃ, ḥ, etc.)
        5. For Tamil text: Preserve all Tamil characters and diacritics accurately
        6. For other Indian language scripts: Maintain authentic character representation
        7. Include any visible verse numbers, section markers, or page headers
        8. Format any tables or structured content appropriately
        9. Ignore watermarks, page numbers, and purely decorative elements
        10. Maintain the logical reading order
        11. If text is unclear or damaged, do your best to reconstruct based on context
        
        IMPORTANT: Always provide a confidence score between 0.0 and 1.0 for extraction accuracy.
        Always detect and specify the primary language of the text.
        If no text is visible or extractable, set extracted_text to an empty string but still provide confidence and language fields.
        """
        
        # Configure for OCR extraction
        config = self.model_config
        config.response_schema = self.ocr_schema
        
        # Convert PIL image to bytes for the API
        import io
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash-preview-05-20',
                contents=[
                    prompt,
                    types.Part.from_bytes(
                        data=img_bytes.getvalue(), 
                        mime_type='image/png'
                    )
                ],
                config=config
            )
            
            # Check if response is valid
            if response is None:
                print(f"Error: API returned None response for page {page_num}")
                return OCRResult(
                    extracted_text=f"[Error: API returned None response for page {page_num}]",
                    confidence=0.0,
                    language_detected="unknown",
                    processing_time=time.time() - start_time
                )
            
            if not hasattr(response, 'text') or response.text is None:
                print(f"Error: API response has no text content for page {page_num}")
                return OCRResult(
                    extracted_text=f"[Error: API response has no text content for page {page_num}]",
                    confidence=0.0,
                    language_detected="unknown",
                    processing_time=time.time() - start_time
                )
            
            # Extract text content from response
            text_content = None
            if hasattr(response, 'text') and response.text:
                text_content = response.text
            elif hasattr(response, 'candidates') and response.candidates:
                # Try alternative access path
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content.parts:
                    part = candidate.content.parts[0]
                    if hasattr(part, 'text'):
                        text_content = part.text
            
            # Use the text content we found
            if not text_content:
                print(f"No text content available for JSON parsing")
                return OCRResult(
                    extracted_text=f"[No text content in API response for page {page_num}]",
                    confidence=0.0,
                    language_detected="unknown",
                    processing_time=time.time() - start_time
                )
            
            try:
                result_data = json.loads(text_content)
            except json.JSONDecodeError as je:
                print(f"JSON parsing error in OCR extraction for page {page_num}: {str(je)}")
                print(f"Problematic response text: {text_content[:500]}...")
                return OCRResult(
                    extracted_text=f"[JSON parsing error: {str(je)}]",
                    confidence=0.0,
                    language_detected="unknown",
                    processing_time=time.time() - start_time
                )
            
            processing_time = time.time() - start_time
            
            # Validate required fields
            extracted_text = result_data.get('extracted_text', '')
            confidence = result_data.get('confidence', 0.0)
            language_detected = result_data.get('language_detected', 'unknown')
            
            print(f"OCR extracted {len(extracted_text)} characters from page {page_num}")
            print(f"OCR confidence: {confidence}, Language: {language_detected}")
            if len(extracted_text) > 100:
                print(f"Text preview: {extracted_text[:100]}...")
            else:
                print(f"Full text: {extracted_text}")
            
            return OCRResult(
                extracted_text=extracted_text,
                confidence=confidence,
                language_detected=language_detected,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Error in OCR extraction for page {page_num}: {str(e)}"
            print(error_msg)
            print(f"Exception type: {type(e)}")
            print(f"Exception details: {repr(e)}")
            
            # Try to get more specific error information
            if hasattr(e, 'args') and e.args:
                print(f"Exception args: {e.args}")
            
            return OCRResult(
                extracted_text=f"[Error extracting text from page {page_num}: {str(e)}]",
                confidence=0.0,
                language_detected="unknown",
                processing_time=processing_time
            )
    
    def clean_and_process_text(self, text: str, book_name: str) -> str:
        """Clean and format the extracted text."""
        # Input validation
        if not text or text.strip() == "":
            print(f"Warning: Empty text provided for cleaning from {book_name}")
            return "[No text content extracted]"
        
        print(f"Cleaning text from {book_name}: {len(text)} characters")
        
        prompt = f"""
        You are tasked with cleaning and formatting OCRed text. The text below was extracted from "{book_name}".
        
        INSTRUCTIONS:
        1. Remove page numbers, headers, footers, and formatting artifacts
        2. Fix and properly format Sanskrit verses and translations
        3. Format chapter headings and sections with appropriate markdown
        4. Preserve diacritical marks in Sanskrit terms
        5. Clean up any table structures with proper markdown formatting
        6. Replace any references to "Paramahamsa Nithyananda" or "Nithyananda" with "THE SUPREME PONTIFF OF HINDUISM BHAGAWAN SRI NITHYANANDA PARAMASHIVAM"
        7. Replace "India" with "Bharat" and "Indian" with "Hindu"
        8. Remove ALL contact information (phones, emails, websites, social media)
        9. Remove ALL book metadata (ISBN, copyright, publication dates, pricing, publisher info)
        10. Remove numerical statistics about humanitarian activities
        11. Maintain the original language - do not translate
        12. Remove unnecessary line breaks and improve flow
        
        IMPORTANT: Return ONLY the cleaned text. Do not add explanations, comments, or metadata.
        If the text appears to be in Tamil, Sanskrit, or another non-English language, preserve it exactly and only clean formatting issues.
        
        TEXT TO CLEAN:
        {text}
        
        CLEANED TEXT:
        """
        
        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash-preview-05-20',
                contents=[prompt],
                config=types.GenerateContentConfig(
                    max_output_tokens=8000,
                    temperature=0.1
                )
            )
            
            # Check if response is valid
            if response is None or not hasattr(response, 'text') or response.text is None:
                print(f"Warning: Text cleaning API returned None response, using original text")
                return text
            
            cleaned_text = response.text.strip()
            
            # Validate the response
            if not cleaned_text or "text to be cleaned was not provided" in cleaned_text.lower() or "please provide" in cleaned_text.lower():
                print(f"Warning: Text cleaning API returned invalid response: '{cleaned_text[:100]}...'")
                print(f"Using original text instead")
                return text
            
            print(f"Text cleaning successful: {len(cleaned_text)} characters")
            return cleaned_text
            
        except Exception as e:
            print(f"Error in text cleaning: {str(e)}")
            return text  # Return original if cleaning fails
    
    def load_progress(self, progress_file: str) -> Dict[int, PageProgress]:
        """Load processing progress from CSV file."""
        progress = {}
        if os.path.exists(progress_file):
            with open(progress_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    page_num = int(row['page_num'])
                    progress[page_num] = PageProgress(
                        page_num=page_num,
                        status=row['status'],
                        legibility_score=float(row['legibility_score']) if row['legibility_score'] else None,
                        semantic_score=float(row['semantic_score']) if row.get('semantic_score') else None,
                        ocr_confidence=float(row['ocr_confidence']) if row['ocr_confidence'] else None,
                        processing_time=float(row['processing_time']),
                        error_message=row['error_message'] if row['error_message'] else None,
                        timestamp=row['timestamp']
                    )
        return progress
    
    def save_progress(self, progress: Dict[int, PageProgress], progress_file: str):
        """Save processing progress to CSV file."""
        fieldnames = ['page_num', 'status', 'legibility_score', 'semantic_score', 'ocr_confidence', 
                     'processing_time', 'error_message', 'timestamp']
        
        with open(progress_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for page_progress in sorted(progress.values(), key=lambda x: x.page_num):
                writer.writerow({
                    'page_num': page_progress.page_num,
                    'status': page_progress.status,
                    'legibility_score': page_progress.legibility_score,
                    'semantic_score': page_progress.semantic_score,
                    'ocr_confidence': page_progress.ocr_confidence,
                    'processing_time': page_progress.processing_time,
                    'error_message': page_progress.error_message,
                    'timestamp': page_progress.timestamp
                })
    
    def process_pdf(self, pdf_path: str, output_dir: str, start_page: int = 1, 
                    end_page: Optional[int] = None, dpi: int = 300, 
                    legibility_threshold: float = 0.5, semantic_threshold: float = 0.6) -> str:
        """
        Process a PDF file with enhanced OCR and legibility detection.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save output
            start_page: First page to process (1-based)
            end_page: Last page to process (inclusive)
            dpi: DPI for rendering PDF pages
            legibility_threshold: Minimum legibility score to attempt OCR
            
        Returns:
            Path to the final processed markdown file
        """
        # Setup output paths
        book_name = Path(pdf_path).stem
        book_output_dir = Path(output_dir) / book_name
        book_output_dir.mkdir(parents=True, exist_ok=True)
        
        progress_file = book_output_dir / f"{book_name}_progress.csv"
        final_output_file = book_output_dir / f"{book_name}_processed.md"
        
        # Load existing progress
        progress = self.load_progress(str(progress_file))
        
        # Open PDF
        print(f"Opening PDF: {pdf_path}")
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        print(f"PDF has {total_pages} pages")
        
        # Adjust page range
        if start_page < 1:
            start_page = 1
        if end_page is None or end_page > total_pages:
            end_page = total_pages
            
        pages_to_process = range(start_page - 1, end_page)  # 0-based index
        num_pages = len(pages_to_process)
        
        print(f"Processing pages {start_page} to {end_page} ({num_pages} pages)")
        print(f"Legibility threshold: {legibility_threshold}")
        print(f"Semantic threshold: {semantic_threshold}")
        print(f"Thinking budget: {self.thinking_budget}")
        
        # Process pages
        all_content = {}
        
        with tqdm(total=num_pages, desc="Processing pages", unit="page") as pbar:
            for i in pages_to_process:
                page_num = i + 1  # Convert to 1-based
                
                # Check if already processed
                if page_num in progress and progress[page_num].status == 'completed':
                    pbar.update(1)
                    pbar.set_description(f"Skipping completed page {page_num}")
                    
                    # Load existing content
                    page_file = book_output_dir / f"page_{page_num:04d}.md"
                    if page_file.exists():
                        with open(page_file, 'r', encoding='utf-8') as f:
                            all_content[page_num] = f.read()
                    continue
                
                try:
                    # Render page as image
                    pbar.set_description(f"Rendering page {page_num}")
                    page = doc[i]
                    pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
                    
                    # Convert to PIL Image
                    mode = "RGBA" if pix.alpha else "RGB"
                    img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
                    
                    # Step 1: Combined pre-assessment (legibility + semantic prediction)
                    pbar.set_description(f"Pre-assessing page {page_num}")
                    assessment_result = self.combined_pre_assessment(
                        img, page_num, legibility_threshold, semantic_threshold
                    )
                    
                    page_content = ""
                    ocr_confidence = None
                    semantic_score = assessment_result.expected_semantic_quality
                    
                    if assessment_result.should_process:
                        # Step 2: Extract text (only if pre-assessment passed)
                        pbar.set_description(f"Extracting text page {page_num}")
                        ocr_result = self.extract_text(img, page_num)
                        
                        # Step 3: Clean text (skip semantic validation since pre-assessed)
                        pbar.set_description(f"Cleaning text page {page_num}")
                        cleaned_text = self.clean_and_process_text(ocr_result.extracted_text, book_name)
                        
                        page_content = f"# Page {page_num}\n\n{cleaned_text}"
                        ocr_confidence = ocr_result.confidence
                        status = 'completed'
                        
                    else:
                        # Mark as not worth processing
                        page_content = f"# Page {page_num} - NOT SUITABLE FOR OCR\n\n**Combined Assessment:**\n"
                        page_content += f"- Legibility Score: {assessment_result.legibility_score:.2f}\n"
                        page_content += f"- Expected Semantic Quality: {assessment_result.expected_semantic_quality:.2f}\n"
                        page_content += f"- Text Clarity: {assessment_result.text_clarity}\n"
                        page_content += f"- Image Quality: {assessment_result.image_quality}\n"
                        page_content += f"- OCR Prediction: {assessment_result.ocr_prediction}\n"
                        page_content += f"- Semantic Prediction: {assessment_result.semantic_prediction}\n"
                        page_content += f"- Visible Text Sample: '{assessment_result.visible_text_sample}'\n"
                        page_content += f"- Language Detected: {assessment_result.language_detected}\n"
                        page_content += f"- Issues Found: {', '.join(assessment_result.issues_found)}\n"
                        page_content += f"- Reason: {assessment_result.reason}"
                        
                        if assessment_result.legibility_score < legibility_threshold:
                            status = 'illegible'
                        else:
                            status = 'semantically_invalid'
                    
                    # Save page content
                    page_file = book_output_dir / f"page_{page_num:04d}.md"
                    with open(page_file, 'w', encoding='utf-8') as f:
                        f.write(page_content)
                    
                    all_content[page_num] = page_content
                    
                    # Update progress
                    total_processing_time = assessment_result.processing_time
                    if 'ocr_result' in locals():
                        total_processing_time += ocr_result.processing_time
                    
                    progress[page_num] = PageProgress(
                        page_num=page_num,
                        status=status,
                        legibility_score=assessment_result.legibility_score,
                        semantic_score=semantic_score,
                        ocr_confidence=ocr_confidence,
                        processing_time=total_processing_time,
                        error_message=None,
                        timestamp=datetime.now().isoformat()
                    )
                    
                    # Save progress after each page
                    self.save_progress(progress, str(progress_file))
                    
                    pbar.update(1)
                    pbar.set_description(f"Completed page {page_num}")
                    
                    # Rate limiting
                    time.sleep(1)
                    
                except Exception as e:
                    error_msg = f"Error processing page {page_num}: {str(e)}"
                    print(f"\n{error_msg}")
                    
                    # Save error page
                    page_content = f"# Page {page_num} - ERROR\n\n{error_msg}"
                    page_file = book_output_dir / f"page_{page_num:04d}.md"
                    with open(page_file, 'w', encoding='utf-8') as f:
                        f.write(page_content)
                    
                    all_content[page_num] = page_content
                    
                    progress[page_num] = PageProgress(
                        page_num=page_num,
                        status='error',
                        legibility_score=None,
                        semantic_score=None,
                        ocr_confidence=None,
                        processing_time=0,
                        error_message=str(e),
                        timestamp=datetime.now().isoformat()
                    )
                    
                    self.save_progress(progress, str(progress_file))
                    pbar.update(1)
        
        doc.close()
        
        # Combine all pages into final markdown file
        print("\nCombining pages into final markdown file...")
        final_content = [f"# {book_name}\n\n"]
        
        for page_num in sorted(all_content.keys()):
            final_content.append(all_content[page_num])
            final_content.append("\n\n---\n\n")  # Page separator
        
        with open(final_output_file, 'w', encoding='utf-8') as f:
            f.write("".join(final_content))
        
        # Print summary
        completed = sum(1 for p in progress.values() if p.status == 'completed')
        illegible = sum(1 for p in progress.values() if p.status == 'illegible')
        semantic_invalid = sum(1 for p in progress.values() if p.status == 'semantically_invalid')
        errors = sum(1 for p in progress.values() if p.status == 'error')
        
        print(f"\nProcessing completed:")
        print(f"- Successful OCR: {completed} pages")
        print(f"- Visually illegible: {illegible} pages")
        print(f"- Semantically invalid: {semantic_invalid} pages")
        print(f"- Error pages: {errors} pages")
        print(f"- Progress saved to: {progress_file}")
        print(f"- Final output: {final_output_file}")
        
        return str(final_output_file)
    
    def process_single_image(self, image_path: str, output_dir: str, 
                           legibility_threshold: float = 0.5, semantic_threshold: float = 0.6) -> str:
        """
        Process a single image file with enhanced OCR and legibility detection.
        
        Args:
            image_path: Path to the image file
            output_dir: Directory to save output
            legibility_threshold: Minimum legibility score to attempt OCR
            
        Returns:
            Path to the processed markdown file
        """
        image_file = Path(image_path)
        if not image_file.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Setup output paths
        image_name = image_file.stem
        output_dir_path = Path(output_dir) / f"{image_name}_image"
        output_dir_path.mkdir(parents=True, exist_ok=True)
        
        progress_file = output_dir_path / f"{image_name}_progress.csv"
        final_output_file = output_dir_path / f"{image_name}_processed.md"
        
        # Check if already processed
        if final_output_file.exists():
            print(f"Image already processed: {final_output_file}")
            return str(final_output_file)
        
        print(f"Processing single image: {image_path}")
        print(f"Legibility threshold: {legibility_threshold}")
        print(f"Semantic threshold: {semantic_threshold}")
        print(f"Thinking budget: {self.thinking_budget}")
        
        try:
            # Load image
            print("Loading image...")
            img = Image.open(image_path)
            
            # Convert to RGB if necessary
            if img.mode not in ('RGB', 'RGBA'):
                img = img.convert('RGB')
            
            # Step 1: Combined pre-assessment
            print("Pre-assessing image quality and content...")
            assessment_result = self.combined_pre_assessment(
                img, 1, legibility_threshold, semantic_threshold
            )
            
            image_content = ""
            ocr_confidence = None
            semantic_score = assessment_result.expected_semantic_quality
            status = ""
            
            if assessment_result.should_process:
                # Step 2: Extract text (only if pre-assessment passed)
                print("Extracting text...")
                ocr_result = self.extract_text(img, 1)
                
                # Step 3: Clean text (skip semantic validation since pre-assessed)
                print("Cleaning text...")
                cleaned_text = self.clean_and_process_text(ocr_result.extracted_text, image_name)
                
                image_content = f"# {image_file.name}\n\n**Source:** `{image_path}`\n\n{cleaned_text}"
                ocr_confidence = ocr_result.confidence
                status = 'completed'
                
                print(f"✅ Text extraction successful (OCR confidence: {ocr_confidence:.2f}, Pre-assessed quality: {semantic_score:.2f})")
                
            else:
                # Mark as not suitable for processing
                image_content = f"# {image_file.name} - NOT SUITABLE FOR OCR\n\n**Source:** `{image_path}`\n\n**Combined Assessment:**\n"
                image_content += f"- Legibility Score: {assessment_result.legibility_score:.2f}\n"
                image_content += f"- Expected Semantic Quality: {assessment_result.expected_semantic_quality:.2f}\n"
                image_content += f"- Text Clarity: {assessment_result.text_clarity}\n"
                image_content += f"- Image Quality: {assessment_result.image_quality}\n"
                image_content += f"- OCR Prediction: {assessment_result.ocr_prediction}\n"
                image_content += f"- Semantic Prediction: {assessment_result.semantic_prediction}\n"
                image_content += f"- Visible Text Sample: '{assessment_result.visible_text_sample}'\n"
                image_content += f"- Language Detected: {assessment_result.language_detected}\n"
                image_content += f"- Issues Found: {', '.join(assessment_result.issues_found)}"
                
                if assessment_result.legibility_score < legibility_threshold:
                    status = 'illegible'
                    print(f"❌ Image marked as illegible (score: {assessment_result.legibility_score:.2f})")
                else:
                    status = 'semantically_invalid'
                    print(f"⚠️ Image has poor expected semantic quality (score: {assessment_result.expected_semantic_quality:.2f})")
            
            # Save the result
            with open(final_output_file, 'w', encoding='utf-8') as f:
                f.write(image_content)
            
            # Calculate total processing time
            total_processing_time = assessment_result.processing_time
            if 'ocr_result' in locals():
                total_processing_time += ocr_result.processing_time
            
            # Save progress record
            progress = {
                image_file.name: ImageProgress(
                    file_path=image_file.name,
                    status=status,
                    legibility_score=assessment_result.legibility_score,
                    semantic_score=semantic_score,
                    ocr_confidence=ocr_confidence,
                    processing_time=total_processing_time,
                    error_message=None,
                    timestamp=datetime.now().isoformat()
                )
            }
            self.save_image_progress(progress, str(progress_file))
            
            print(f"\nProcessing completed:")
            print(f"- Status: {status}")
            print(f"- Legibility score: {assessment_result.legibility_score:.2f}")
            if semantic_score:
                print(f"- Expected semantic quality: {semantic_score:.2f}")
            if ocr_confidence:
                print(f"- OCR confidence: {ocr_confidence:.2f}")
            print(f"- Progress saved to: {progress_file}")
            print(f"- Output file: {final_output_file}")
            
            return str(final_output_file)
            
        except Exception as e:
            error_msg = f"Error processing {image_path}: {str(e)}"
            print(f"❌ {error_msg}")
            
            # Save error content
            error_content = f"# {image_file.name} - ERROR\n\n**Source:** `{image_path}`\n\n{error_msg}"
            with open(final_output_file, 'w', encoding='utf-8') as f:
                f.write(error_content)
            
            # Save error progress
            progress = {
                image_file.name: ImageProgress(
                    file_path=image_file.name,
                    status='error',
                    legibility_score=None,
                    semantic_score=None,
                    ocr_confidence=None,
                    processing_time=0,
                    error_message=str(e),
                    timestamp=datetime.now().isoformat()
                )
            }
            self.save_image_progress(progress, str(progress_file))
            
            return str(final_output_file)
    
    def is_pdf_file(self, file_path: str) -> bool:
        """Check if the file is a PDF."""
        return Path(file_path).suffix.lower() == '.pdf'
    
    def is_image_file(self, file_path: str) -> bool:
        """Check if the file is a supported image format."""
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        return Path(file_path).suffix.lower() in supported_formats
    
    def find_image_files(self, input_dir: str) -> List[Tuple[str, str]]:
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
    
    def process_images(self, input_dir: str, output_dir: str, 
                      legibility_threshold: float = 0.5, semantic_threshold: float = 0.6) -> str:
        """
        Process all image files in a directory with enhanced OCR and legibility detection.
        
        Args:
            input_dir: Directory containing image files
            output_dir: Directory to save output
            legibility_threshold: Minimum legibility score to attempt OCR
            
        Returns:
            Path to the final processed markdown file
        """
        # Find all image files
        print(f"Scanning for image files in: {input_dir}")
        image_files = self.find_image_files(input_dir)
        
        if not image_files:
            print("No supported image files found!")
            return ""
        
        print(f"Found {len(image_files)} image files")
        
        # Setup output paths
        dir_name = Path(input_dir).name
        output_dir_path = Path(output_dir) / f"{dir_name}_images"
        output_dir_path.mkdir(parents=True, exist_ok=True)
        
        progress_file = output_dir_path / f"{dir_name}_images_progress.csv"
        final_output_file = output_dir_path / f"{dir_name}_images_processed.md"
        
        # Load existing progress (using file path as key instead of page number)
        progress = self.load_image_progress(str(progress_file))
        
        print(f"Processing {len(image_files)} images")
        print(f"Legibility threshold: {legibility_threshold}")
        print(f"Semantic threshold: {semantic_threshold}")
        print(f"Thinking budget: {self.thinking_budget}")
        
        # Process images
        all_content = {}
        
        with tqdm(total=len(image_files), desc="Processing images", unit="image") as pbar:
            for idx, (file_path, relative_path) in enumerate(image_files, 1):
                
                # Check if already processed
                if relative_path in progress and progress[relative_path].status == 'completed':
                    pbar.update(1)
                    pbar.set_description(f"Skipping completed: {Path(relative_path).name}")
                    
                    # Load existing content
                    safe_filename = self.get_safe_filename(relative_path)
                    image_file = output_dir_path / f"{safe_filename}.md"
                    if image_file.exists():
                        with open(image_file, 'r', encoding='utf-8') as f:
                            all_content[relative_path] = f.read()
                    continue
                
                try:
                    # Load image
                    pbar.set_description(f"Loading: {Path(relative_path).name}")
                    img = Image.open(file_path)
                    
                    # Convert to RGB if necessary
                    if img.mode not in ('RGB', 'RGBA'):
                        img = img.convert('RGB')
                    
                    # Step 1: Combined pre-assessment
                    pbar.set_description(f"Pre-assessing: {Path(relative_path).name}")
                    assessment_result = self.combined_pre_assessment(
                        img, idx, legibility_threshold, semantic_threshold
                    )
                    
                    image_content = ""
                    ocr_confidence = None
                    semantic_score = assessment_result.expected_semantic_quality
                    
                    if assessment_result.should_process:
                        # Step 2: Extract text (only if pre-assessment passed)
                        pbar.set_description(f"Extracting text: {Path(relative_path).name}")
                        ocr_result = self.extract_text(img, idx)
                        
                        # Step 3: Clean text (skip semantic validation since pre-assessed)
                        pbar.set_description(f"Cleaning text: {Path(relative_path).name}")
                        cleaned_text = self.clean_and_process_text(ocr_result.extracted_text, dir_name)
                        
                        image_content = f"# {relative_path}\n\n**Source:** `{file_path}`\n\n{cleaned_text}"
                        ocr_confidence = ocr_result.confidence
                        status = 'completed'
                        
                    else:
                        # Mark as not suitable for processing
                        image_content = f"# {relative_path} - NOT SUITABLE FOR OCR\n\n**Source:** `{file_path}`\n\n**Combined Assessment:**\n"
                        image_content += f"- Legibility Score: {assessment_result.legibility_score:.2f}\n"
                        image_content += f"- Expected Semantic Quality: {assessment_result.expected_semantic_quality:.2f}\n"
                        image_content += f"- Text Clarity: {assessment_result.text_clarity}\n"
                        image_content += f"- Image Quality: {assessment_result.image_quality}\n"
                        image_content += f"- Visible Text Sample: '{assessment_result.visible_text_sample}'\n"
                        image_content += f"- Language Detected: {assessment_result.language_detected}\n"
                        image_content += f"- Issues Found: {', '.join(assessment_result.issues_found)}\n"
                        image_content += f"- Reason: {assessment_result.reason}"
                        
                        if assessment_result.legibility_score < legibility_threshold:
                            status = 'illegible'
                        else:
                            status = 'semantically_invalid'
                    
                    # Save individual image content
                    safe_filename = self.get_safe_filename(relative_path)
                    image_file = output_dir_path / f"{safe_filename}.md"
                    with open(image_file, 'w', encoding='utf-8') as f:
                        f.write(image_content)
                    
                    all_content[relative_path] = image_content
                    
                    # Calculate total processing time
                    total_processing_time = assessment_result.processing_time
                    if 'ocr_result' in locals():
                        total_processing_time += ocr_result.processing_time
                    
                    # Update progress
                    progress[relative_path] = ImageProgress(
                        file_path=relative_path,
                        status=status,
                        legibility_score=assessment_result.legibility_score,
                        semantic_score=semantic_score,
                        ocr_confidence=ocr_confidence,
                        processing_time=total_processing_time,
                        error_message=None,
                        timestamp=datetime.now().isoformat()
                    )
                    
                    # Save progress after each image
                    self.save_image_progress(progress, str(progress_file))
                    
                    pbar.update(1)
                    pbar.set_description(f"Completed: {Path(relative_path).name}")
                    
                    # Rate limiting
                    time.sleep(1)
                    
                except Exception as e:
                    error_msg = f"Error processing {relative_path}: {str(e)}"
                    print(f"\n{error_msg}")
                    
                    # Save error content
                    image_content = f"# {relative_path} - ERROR\n\n**Source:** `{file_path}`\n\n{error_msg}"
                    safe_filename = self.get_safe_filename(relative_path)
                    image_file = output_dir_path / f"{safe_filename}.md"
                    with open(image_file, 'w', encoding='utf-8') as f:
                        f.write(image_content)
                    
                    all_content[relative_path] = image_content
                    
                    progress[relative_path] = ImageProgress(
                        file_path=relative_path,
                        status='error',
                        legibility_score=None,
                        semantic_score=None,
                        ocr_confidence=None,
                        processing_time=0,
                        error_message=str(e),
                        timestamp=datetime.now().isoformat()
                    )
                    
                    self.save_image_progress(progress, str(progress_file))
                    pbar.update(1)
        
        # Combine all images into final markdown file
        print("\nCombining images into final markdown file...")
        final_content = [f"# {dir_name} - Image OCR Results\n\n"]
        
        # Group by subdirectory for better organization
        grouped_content = {}
        for relative_path in sorted(all_content.keys()):
            subdir = str(Path(relative_path).parent) if Path(relative_path).parent != Path('.') else 'root'
            if subdir not in grouped_content:
                grouped_content[subdir] = []
            grouped_content[subdir].append((relative_path, all_content[relative_path]))
        
        for subdir in sorted(grouped_content.keys()):
            if subdir != 'root':
                final_content.append(f"## {subdir}\n\n")
            
            for relative_path, content in grouped_content[subdir]:
                final_content.append(content)
                final_content.append("\n\n---\n\n")  # Content separator
        
        with open(final_output_file, 'w', encoding='utf-8') as f:
            f.write("".join(final_content))
        
        # Print summary
        completed = sum(1 for p in progress.values() if p.status == 'completed')
        illegible = sum(1 for p in progress.values() if p.status == 'illegible')
        semantic_invalid = sum(1 for p in progress.values() if p.status == 'semantically_invalid')
        errors = sum(1 for p in progress.values() if p.status == 'error')
        
        print(f"\nProcessing completed:")
        print(f"- Successful OCR: {completed} images")
        print(f"- Visually illegible: {illegible} images")
        print(f"- Semantically invalid: {semantic_invalid} images")
        print(f"- Error images: {errors} images")
        print(f"- Progress saved to: {progress_file}")
        print(f"- Final output: {final_output_file}")
        
        return str(final_output_file)
    
    def get_safe_filename(self, relative_path: str) -> str:
        """Convert relative path to safe filename for markdown files."""
        # Replace path separators and special characters
        safe_name = str(relative_path).replace('/', '_').replace('\\', '_')
        safe_name = re.sub(r'[<>:"|?*]', '_', safe_name)
        # Remove file extension and add index if too long
        safe_name = Path(safe_name).stem
        if len(safe_name) > 100:
            safe_name = safe_name[:100]
        return safe_name
    
    def load_image_progress(self, progress_file: str) -> Dict[str, 'ImageProgress']:
        """Load image processing progress from CSV file."""
        progress = {}
        if os.path.exists(progress_file):
            with open(progress_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    file_path = row['file_path']
                    progress[file_path] = ImageProgress(
                        file_path=file_path,
                        status=row['status'],
                        legibility_score=float(row['legibility_score']) if row['legibility_score'] else None,
                        semantic_score=float(row['semantic_score']) if row.get('semantic_score') else None,
                        ocr_confidence=float(row['ocr_confidence']) if row['ocr_confidence'] else None,
                        processing_time=float(row['processing_time']),
                        error_message=row['error_message'] if row['error_message'] else None,
                        timestamp=row['timestamp']
                    )
        return progress
    
    def save_image_progress(self, progress: Dict[str, 'ImageProgress'], progress_file: str):
        """Save image processing progress to CSV file."""
        fieldnames = ['file_path', 'status', 'legibility_score', 'semantic_score', 'ocr_confidence', 
                     'processing_time', 'error_message', 'timestamp']
        
        with open(progress_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for image_progress in sorted(progress.values(), key=lambda x: x.file_path):
                writer.writerow({
                    'file_path': image_progress.file_path,
                    'status': image_progress.status,
                    'legibility_score': image_progress.legibility_score,
                    'semantic_score': image_progress.semantic_score,
                    'ocr_confidence': image_progress.ocr_confidence,
                    'processing_time': image_progress.processing_time,
                    'error_message': image_progress.error_message,
                    'timestamp': image_progress.timestamp
                })


@dataclass
class ImageProgress:
    file_path: str
    status: str  # 'pending', 'visually_legible', 'illegible', 'semantically_invalid', 'completed', 'error'
    legibility_score: Optional[float]
    semantic_score: Optional[float]
    ocr_confidence: Optional[float]
    processing_time: float
    error_message: Optional[str]
    timestamp: str


def main():
    parser = argparse.ArgumentParser(description="Enhanced OCR with Gemini 2.5 Flash and legibility detection")
    
    # Input options - PDF file, image file, or image directory
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-file", help="PDF file or single image file to process")
    input_group.add_argument("--input-dir", help="Directory containing image files")
    
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--api-key", required=True, help="Gemini API key")
    
    # PDF-specific options
    parser.add_argument("--start-page", type=int, default=1, help="Start page for PDF (1-based)")
    parser.add_argument("--end-page", type=int, help="End page for PDF (inclusive)")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for PDF rendering")
    
    # Common options
    parser.add_argument("--legibility-threshold", type=float, default=0.5, help="Minimum legibility score (0-1)")
    parser.add_argument("--semantic-threshold", type=float, default=0.6, help="Minimum semantic meaningfulness score (0-1)")
    parser.add_argument("--thinking-budget", type=int, default=2000, help="Thinking budget for Gemini")
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.input_file and not os.path.exists(args.input_file):
        print(f"Error: File not found: {args.input_file}")
        return
    
    if args.input_dir and not os.path.exists(args.input_dir):
        print(f"Error: Directory not found: {args.input_dir}")
        return
    
    # Create OCR processor
    ocr = GeminiAdvancedOCR(args.api_key, thinking_budget=args.thinking_budget)
    
    # Process based on input type
    if args.input_file:
        if ocr.is_pdf_file(args.input_file):
            # Process as PDF
            print(f"\n{'='*80}\nProcessing PDF: {args.input_file}\n{'='*80}")
            result_file = ocr.process_pdf(
                args.input_file,
                args.output_dir,
                args.start_page,
                args.end_page,
                args.dpi,
                args.legibility_threshold,
                args.semantic_threshold
            )
            print(f"\nPDF processing complete! Final file: {result_file}")
            
        elif ocr.is_image_file(args.input_file):
            # Process as single image
            print(f"\n{'='*80}\nProcessing single image: {args.input_file}\n{'='*80}")
            result_file = ocr.process_single_image(
                args.input_file,
                args.output_dir,
                args.legibility_threshold,
                args.semantic_threshold
            )
            print(f"\nImage processing complete! Final file: {result_file}")
            
        else:
            print(f"Error: Unsupported file format: {args.input_file}")
            print("Supported formats: PDF, JPG, JPEG, PNG, BMP, TIFF, TIF, WEBP")
            return
        
    elif args.input_dir:
        print(f"\n{'='*80}\nProcessing image directory: {args.input_dir}\n{'='*80}")
        result_file = ocr.process_images(
            args.input_dir,
            args.output_dir,
            args.legibility_threshold,
            args.semantic_threshold
        )
        print(f"\nImage directory processing complete! Final file: {result_file}")


if __name__ == "__main__":
    main()