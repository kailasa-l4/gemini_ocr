"""
Semantic validation engine for determining if extracted text is meaningful.

This module validates whether OCR-extracted text contains coherent,
semantically meaningful content or is just gibberish/random characters.
"""

import time
import logging
from typing import Optional

from ..api import GeminiAPIClient
from ..prompts import PromptTemplateManager
from ..utils.json_recovery import JSONRecoveryService
from ..models import SemanticResult
from ..schemas import APISchemas


class SemanticValidator:
    """
    Specialized engine for validating semantic meaning of extracted text.
    
    Analyzes OCR output to determine if it contains meaningful, coherent
    text or is primarily gibberish that should be discarded.
    """
    
    def __init__(self, 
                 api_client: GeminiAPIClient,
                 prompt_manager: PromptTemplateManager,
                 json_recovery: JSONRecoveryService,
                 logger: Optional[logging.Logger] = None,
                 model: str = 'gemini-2.5-flash'):
        """
        Initialize the semantic validator.
        
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
        self.logger = logger or logging.getLogger('SemanticValidator')
    
    def validate_semantic_meaning(self, text: str, language: str, page_num: int) -> SemanticResult:
        """
        Validate if the extracted text is semantically meaningful.
        
        Args:
            text: Extracted text to validate
            language: Detected language of the text
            page_num: Page number for logging context
            
        Returns:
            SemanticResult with validation details
        """
        start_time = time.time()
        
        try:
            # Get prompt from template manager
            prompt = self.prompt_manager.get_prompt(
                'semantic_validation',
                text=text,
                language=language
            )
            
            # Configure for semantic validation
            config = self.api_client.create_config(
                response_schema=APISchemas.get_semantic_schema()
            )
            
            # Make API call
            response = self.api_client.generate_content(
                prompt=prompt,
                config=config,
                model=self.model,
                context=f"semantic validation page {page_num}"
            )
            
            # Parse response with recovery support
            result_data = self.json_recovery.validate_and_recover(
                response.text,
                context=f"semantic validation page {page_num}"
            )
            
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
            error_msg = f"Error in semantic validation for page {page_num}: {str(e)}"
            self.logger.error(error_msg)
            
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