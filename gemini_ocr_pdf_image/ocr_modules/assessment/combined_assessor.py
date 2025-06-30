"""
Combined assessment engine for comprehensive OCR viability evaluation.

This module performs integrated legibility and semantic quality prediction
in a single API call to efficiently determine if OCR should be attempted.
"""

import time
import logging
from typing import Optional
from PIL import Image
from google.genai import types

from ..api import GeminiAPIClient
from ..prompts import PromptTemplateManager
from ..utils.json_recovery import JSONRecoveryService
from ..models import CombinedAssessmentResult
from ..schemas import APISchemas


class CombinedAssessor:
    """
    Specialized engine for combined legibility and semantic pre-assessment.
    
    Performs comprehensive evaluation in a single API call to determine
    if an image is worth processing with full OCR extraction.
    """
    
    def __init__(self, 
                 api_client: GeminiAPIClient,
                 prompt_manager: PromptTemplateManager,
                 json_recovery: JSONRecoveryService,
                 thinking_budget: int = 2000,
                 enable_thinking: bool = True,
                 logger: Optional[logging.Logger] = None,
                 model: str = 'gemini-2.5-flash'):
        """
        Initialize the combined assessor.
        
        Args:
            api_client: Gemini API client for making requests
            prompt_manager: Template manager for prompts
            json_recovery: JSON recovery service for error handling
            thinking_budget: Token budget for thinking process
            enable_thinking: Whether to enable thinking for assessment
            logger: Optional logger instance
        """
        self.api_client = api_client
        self.prompt_manager = prompt_manager
        self.json_recovery = json_recovery
        self.thinking_budget = thinking_budget
        self.enable_thinking = enable_thinking
        self.model = model
        self.logger = logger or logging.getLogger('CombinedAssessor')
    
    def combined_pre_assessment(self, 
                              image: Image.Image, 
                              page_num: int,
                              legibility_threshold: float = 0.5,
                              semantic_threshold: float = 0.6) -> CombinedAssessmentResult:
        """
        Combined legibility and semantic pre-assessment to determine if OCR should be attempted.
        
        Args:
            image: PIL Image to assess
            page_num: Page number for logging context
            legibility_threshold: Minimum legibility score required
            semantic_threshold: Minimum semantic quality score required
            
        Returns:
            CombinedAssessmentResult with comprehensive assessment
        """
        start_time = time.time()
        
        try:
            # Get prompt from template manager
            prompt = self.prompt_manager.get_prompt(
                'combined_assessment',
                page_num=page_num,
                legibility_threshold=legibility_threshold,
                semantic_threshold=semantic_threshold
            )
            
            # Create configuration with thinking support if enabled
            thinking_config = None
            if self.enable_thinking:
                try:
                    thinking_config = types.ThinkingConfig(thinking_budget=self.thinking_budget)
                    self.logger.debug(f"Applied thinking config for assessment with budget: {self.thinking_budget}")
                except Exception as e:
                    self.logger.warning(f"Could not apply thinking config for assessment: {str(e)}")
            
            config = self.api_client.create_config(
                response_schema=APISchemas.get_combined_assessment_schema(),
                max_output_tokens=4000,
                thinking_config=thinking_config
            )
            
            # Make API call
            response = self.api_client.generate_content(
                prompt=prompt,
                image=image,
                config=config,
                model=self.model,
                context=f"combined assessment page {page_num}"
            )
            
            # Parse response with recovery support
            result_data = self.json_recovery.validate_and_recover(
                response.text,
                context=f"combined assessment page {page_num}"
            )
            
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
            error_msg = f"Error in combined assessment for page {page_num}: {str(e)}"
            self.logger.error(error_msg)
            
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