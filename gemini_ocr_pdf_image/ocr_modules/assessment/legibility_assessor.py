"""
Legibility assessment engine for determining if images have readable text.

This module handles visual clarity assessment to determine if an image
contains text that is clear enough for successful OCR processing.
"""

import time
import logging
from typing import Optional
from PIL import Image

from ..api import GeminiAPIClient
from ..prompts import PromptTemplateManager
from ..utils.json_recovery import JSONRecoveryService
from ..models import LegibilityResult
from ..schemas import APISchemas


class LegibilityAssessor:
    """
    Specialized engine for assessing visual legibility of text in images.
    
    Determines if text in an image is clear enough for OCR processing
    by evaluating factors like image quality, text clarity, and readability.
    """
    
    def __init__(self, 
                 api_client: GeminiAPIClient,
                 prompt_manager: PromptTemplateManager,
                 json_recovery: JSONRecoveryService,
                 logger: Optional[logging.Logger] = None,
                 model: str = 'gemini-2.5-flash'):
        """
        Initialize the legibility assessor.
        
        Args:
            api_client: Gemini API client for making requests
            prompt_manager: Template manager for prompts
            json_recovery: JSON recovery service for error handling
            logger: Optional logger instance
        """
        self.api_client = api_client
        self.prompt_manager = prompt_manager
        self.json_recovery = json_recovery
        self.model = model
        self.logger = logger or logging.getLogger('LegibilityAssessor')
    
    def assess_legibility(self, image: Image.Image, page_num: int) -> LegibilityResult:
        """
        Assess if the image text is legible enough for OCR processing.
        
        Args:
            image: PIL Image to assess
            page_num: Page number for logging context
            
        Returns:
            LegibilityResult with assessment details
        """
        start_time = time.time()
        
        try:
            # Get prompt from template manager
            prompt = self.prompt_manager.get_prompt('legibility_assessment', page_num=page_num)
            
            # Configure for legibility assessment
            config = self.api_client.create_config(
                response_schema=APISchemas.get_legibility_schema()
            )
            
            # Make API call
            response = self.api_client.generate_content(
                prompt=prompt,
                image=image,
                config=config,
                model=self.model,
                context=f"legibility assessment page {page_num}"
            )
            
            # Parse response with recovery support
            result_data = self.json_recovery.validate_and_recover(
                response.text,
                context=f"legibility assessment page {page_num}"
            )
            
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
            error_msg = f"Error in legibility assessment for page {page_num}: {str(e)}"
            self.logger.error(error_msg)
            
            return LegibilityResult(
                is_legible=False,
                confidence_score=0.0,
                text_clarity="poor",
                image_quality="poor",
                reason=f"Error during assessment: {str(e)}",
                processing_time=processing_time
            )