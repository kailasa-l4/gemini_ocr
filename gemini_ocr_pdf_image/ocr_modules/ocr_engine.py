"""
Core OCR engine with Gemini API integration.
"""

import io
import json
import logging
import os
import time
from datetime import datetime
from PIL import Image
from google import genai
from google.genai import types
from .models import LegibilityResult, CombinedAssessmentResult, OCRResult, SemanticResult
from .schemas import APISchemas
from .api import GeminiAPIClient
from .prompts import PromptTemplateManager
from .utils.json_recovery import JSONRecoveryService
from .assessment import LegibilityAssessor, SemanticValidator, CombinedAssessor


class GeminiOCREngine:
    """Core OCR processing engine using Gemini 2.5 Flash."""
    
    def __init__(self, api_key: str, thinking_budget: int = 2000, 
                 enable_thinking_assessment: bool = True, enable_thinking_ocr: bool = False,
                 db_logger=None, logs_dir: str = './logs', verbose: bool = False,
                 assessment_model: str = 'gemini-2.5-flash', ocr_model: str = 'gemini-2.5-flash'):
        """Initialize the OCR engine with modular services."""
        self.api_key = api_key
        self.thinking_budget = thinking_budget
        self.enable_thinking_assessment = enable_thinking_assessment
        self.enable_thinking_ocr = enable_thinking_ocr
        self.db_logger = db_logger
        self.logs_dir = logs_dir
        self.verbose = verbose
        self.assessment_model = assessment_model
        self.ocr_model = ocr_model
        
        # Setup logging first
        self.setup_logging()
        
        # Initialize core services
        self.api_client = GeminiAPIClient(api_key, logger=self.logger)
        self.prompt_manager = PromptTemplateManager()
        self.json_recovery = JSONRecoveryService(logger=self.logger)
        
        # Initialize specialized assessment services
        self.legibility_assessor = LegibilityAssessor(
            self.api_client, self.prompt_manager, self.json_recovery, self.logger, self.assessment_model
        )
        self.semantic_validator = SemanticValidator(
            self.api_client, self.prompt_manager, self.json_recovery, self.logger, self.assessment_model
        )
        self.combined_assessor = CombinedAssessor(
            self.api_client, self.prompt_manager, self.json_recovery,
            thinking_budget, enable_thinking_assessment, self.logger, self.assessment_model
        )
        
        # Legacy compatibility - keep for now
        self.client = self.api_client.client
        self.model_config = self.api_client.default_config
        
        if self.verbose:
            print(f"Initialized OCR engine with thinking budget: {thinking_budget}")
            print(f"Thinking enabled - Assessment: {enable_thinking_assessment}, OCR: {enable_thinking_ocr}")
        self.logger.info(f"OCR engine initialized with thinking budget: {thinking_budget}")
        self.logger.info(f"Thinking enabled - Assessment: {enable_thinking_assessment}, OCR: {enable_thinking_ocr}")
        self.logger.info(f"Thinking enabled - Assessment: {enable_thinking_assessment}, OCR: {enable_thinking_ocr}")
    
    def setup_logging(self):
        """Setup logging configuration with file output."""
        # Create logs directory if it doesn't exist
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        
        # Setup logger
        self.logger = logging.getLogger('GeminiOCREngine')
        self.logger.setLevel(logging.DEBUG)
        
        # Create file handler with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.logs_dir, f"ocr_debug_{timestamp}.log")
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
        
        self.logger.info(f"Logging initialized - log file: {log_file}")
        print(f"Debug logging enabled - log file: {log_file}")
    
    def combined_pre_assessment(self, image: Image.Image, page_num: int, 
                              legibility_threshold: float = 0.5, 
                              semantic_threshold: float = 0.6) -> CombinedAssessmentResult:
        """Combined legibility and semantic pre-assessment to determine if OCR should be attempted."""
        return self.combined_assessor.combined_pre_assessment(
            image, page_num, legibility_threshold, semantic_threshold
        )
    
    def assess_legibility(self, image: Image.Image, page_num: int) -> LegibilityResult:
        """Assess if the image text is legible enough for OCR processing."""
        return self.legibility_assessor.assess_legibility(image, page_num)
    
    def extract_text(self, image: Image.Image, page_num: int) -> OCRResult:
        """Extract text from a legible image using OCR."""
        start_time = time.time()
        
        try:
            # Get prompt from template manager
            prompt = self.prompt_manager.get_prompt(
                'ocr_extraction',
                page_num=page_num
            )
            
            # Create configuration with thinking support if enabled
            thinking_config = None
            if self.enable_thinking_ocr:
                try:
                    thinking_config = types.ThinkingConfig(thinking_budget=self.thinking_budget)
                    if self.verbose:
                        self.logger.info(f"Applied thinking config for OCR with budget: {self.thinking_budget}")
                except Exception as e:
                    if self.verbose:
                        self.logger.warning(f"Could not apply thinking config for OCR: {str(e)}")
            else:
                if self.verbose:
                    self.logger.info("Thinking disabled for OCR phase")
            
            config = self.api_client.create_config(
                response_schema=APISchemas.get_ocr_schema(),
                max_output_tokens=6000,  # Increased to handle larger tables
                thinking_config=thinking_config
            )
            
            # Use the API client service with built-in retry logic
            response = self.api_client.generate_content(
                prompt=prompt,
                image=image,
                config=config,
                model=self.ocr_model,
                max_retries=3,
                context=f"OCR extraction page {page_num}"
            )
            
            # Parse JSON with recovery support
            result_data = self.json_recovery.validate_and_recover(
                response.text,
                context=f"OCR extraction page {page_num}"
            )
            
            processing_time = time.time() - start_time
            
            # Validate required fields
            extracted_text = result_data.get('extracted_text', '')
            confidence = result_data.get('confidence', 0.0)
            language_detected = result_data.get('language_detected', 'unknown')
            
            if self.verbose:
                self.logger.info(f"OCR extracted {len(extracted_text)} characters from page {page_num}")
                self.logger.info(f"OCR confidence: {confidence}, Language: {language_detected}")
                if len(extracted_text) > 100:
                    self.logger.info(f"Text preview: {extracted_text[:100]}...")
                else:
                    self.logger.info(f"Full text: {extracted_text}")
            
            return OCRResult(
                extracted_text=extracted_text,
                confidence=confidence,
                language_detected=language_detected,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Error in OCR extraction for page {page_num}: {str(e)}"
            self.logger.error(error_msg)
            if self.verbose:
                self.logger.error(f"Exception type: {type(e)}")
                self.logger.error(f"Exception details: {repr(e)}")
                self.logger.error(f"Exception args: {e.args}")
            
            # Enhanced error logging
            self.logger.error(f"OCR extraction failed for page {page_num}")
            self.logger.error(f"Exception type: {type(e)}")
            self.logger.error(f"Exception message: {str(e)}")
            self.logger.error(f"Exception details: {repr(e)}")
            self.logger.error(f"Exception args: {e.args}")
            self.logger.error(f"Processing time before error: {processing_time:.2f}s")
            
            # Log additional debugging info if available
            if 'response' in locals():
                self.logger.error(f"Response object exists: {response is not None}")
                if response:
                    self.logger.error(f"Response type at error: {type(response)}")
                    try:
                        self.logger.error(f"Response attributes at error: {dir(response)}")
                    except Exception as attr_e:
                        self.logger.error(f"Could not get response attributes: {str(attr_e)}")
            
            return OCRResult(
                extracted_text=f"[Error extracting text from page {page_num}: {str(e)}]",
                confidence=0.0,
                language_detected="unknown",
                processing_time=processing_time
            )
    
    def validate_semantic_meaning(self, text: str, language: str, page_num: int) -> SemanticResult:
        """Validate if the extracted text is semantically meaningful."""
        return self.semantic_validator.validate_semantic_meaning(text, language, page_num)
