"""
Gemini API client for OCR operations.

This module provides a clean interface to the Google Gemini API,
handling configuration, retries, and response validation.
"""

import io
import time
import logging
from typing import Optional, Union
from PIL import Image
from google import genai
from google.genai import types


class GeminiAPIClient:
    """
    Client for Google Gemini API with built-in retry logic and error handling.
    
    Handles:
    - API client configuration and initialization
    - Retry logic with exponential backoff
    - Response validation and error handling
    - Rate limiting and throttling
    """
    
    def __init__(self, api_key: str, logger: Optional[logging.Logger] = None):
        """
        Initialize the Gemini API client.
        
        Args:
            api_key: Google Gemini API key
            logger: Optional logger instance for debugging
        """
        self.api_key = api_key
        self.logger = logger or logging.getLogger('GeminiAPIClient')
        
        # Initialize the Google Gen AI client
        self.client = genai.Client(api_key=api_key)
        
        # Default model configuration
        self.default_config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=None,  # Will be set per request
            max_output_tokens=6000,
            temperature=0.1
        )
        
        self.logger.info("Gemini API client initialized")
    
    def generate_content(
        self,
        prompt: str,
        image: Optional[Image.Image] = None,
        config: Optional[types.GenerateContentConfig] = None,
        model: str = "gemini-2.5-flash",
        max_retries: int = 3,
        initial_delay: float = 2.0,
        context: str = "operation"
    ) -> types.GenerateContentResponse:
        """
        Generate content using Gemini API with retry logic.
        
        Args:
            prompt: Text prompt for the model
            image: Optional PIL Image to include
            config: Optional configuration override
            model: Gemini model to use (default: gemini-2.5-flash)
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay between retries (seconds)
            context: Context description for logging
            
        Returns:
            GenerateContentResponse from Gemini API
            
        Raises:
            Exception: If all retry attempts fail
        """
        # Use provided config or default
        api_config = config or self.default_config
        
        # Prepare content list
        contents = [prompt]
        
        # Add image if provided
        if image:
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            contents.append(
                types.Part.from_bytes(
                    data=img_bytes.getvalue(),
                    mime_type='image/png'
                )
            )
        
        # Retry logic with exponential backoff
        retry_delay = initial_delay
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                self.logger.debug(f"API call attempt {attempt + 1}/{max_retries} for {context}")
                
                response = self.client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=api_config
                )
                
                # Validate response
                self._validate_response(response, context)
                
                self.logger.debug(f"API call succeeded on attempt {attempt + 1} for {context}")
                return response
                
            except Exception as e:
                last_exception = e
                self.logger.warning(
                    f"API call attempt {attempt + 1} failed for {context}: {str(e)}"
                )
                
                if attempt == max_retries - 1:
                    # Last attempt failed
                    self.logger.error(f"All {max_retries} API attempts failed for {context}")
                    break
                else:
                    # Wait before retrying
                    self.logger.info(f"Retrying API call in {retry_delay:.1f} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
        
        # All retries failed
        raise Exception(f"API call failed after {max_retries} attempts: {str(last_exception)}")
    
    def _validate_response(
        self,
        response: types.GenerateContentResponse,
        context: str
    ) -> None:
        """
        Validate API response and log debugging information.
        
        Args:
            response: Response from Gemini API
            context: Context description for logging
            
        Raises:
            Exception: If response is invalid
        """
        self.logger.debug(f"API response received for {context}")
        self.logger.debug(f"Response type: {type(response)}")
        
        if response is None:
            raise Exception(f"API returned None response for {context}")
        
        # Check for text content
        has_text = hasattr(response, 'text') and response.text is not None
        self.logger.debug(f"Response has text: {has_text}")
        
        if has_text:
            self.logger.debug(f"Response text length: {len(response.text)}")
            self.logger.debug(f"Response text preview: {response.text[:200]}...")
        
        # Check for candidates (alternative access path)
        has_candidates = hasattr(response, 'candidates') and response.candidates
        self.logger.debug(f"Response has candidates: {has_candidates}")
        
        if has_candidates:
            self.logger.debug(f"Number of candidates: {len(response.candidates)}")
        
        # Check for MAX_TOKENS issue
        if has_candidates and not has_text:
            candidate = response.candidates[0]
            if hasattr(candidate, 'finish_reason'):
                finish_reason = str(candidate.finish_reason)
                if 'MAX_TOKENS' in finish_reason:
                    self.logger.warning(f"MAX_TOKENS detected for {context}")
                    raise Exception(f"Response truncated due to MAX_TOKENS for {context}")
        
        # Ensure we have some way to access content
        if not has_text and not has_candidates:
            raise Exception(f"API response has no accessible content for {context}")
    
    def create_config(
        self,
        response_schema: Optional[dict] = None,
        max_output_tokens: int = 6000,
        temperature: float = 0.1,
        thinking_config: Optional[types.ThinkingConfig] = None
    ) -> types.GenerateContentConfig:
        """
        Create a configuration for API calls.
        
        Args:
            response_schema: JSON schema for response validation
            max_output_tokens: Maximum tokens in response
            temperature: Sampling temperature
            thinking_config: Optional thinking configuration
            
        Returns:
            GenerateContentConfig instance
        """
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=response_schema,
            max_output_tokens=max_output_tokens,
            temperature=temperature
        )
        
        if thinking_config:
            config.thinking_config = thinking_config
        
        return config