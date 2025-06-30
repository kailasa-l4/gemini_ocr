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


class GeminiOCREngine:
    """Core OCR processing engine using Gemini 2.5 Flash."""
    
    def __init__(self, api_key: str, thinking_budget: int = 2000, 
                 enable_thinking_assessment: bool = True, enable_thinking_ocr: bool = False,
                 db_logger=None, logs_dir: str = './logs'):
        """Initialize the OCR engine with Google Gen AI SDK."""
        self.api_key = api_key
        self.thinking_budget = thinking_budget
        self.enable_thinking_assessment = enable_thinking_assessment
        self.enable_thinking_ocr = enable_thinking_ocr
        self.db_logger = db_logger
        self.logs_dir = logs_dir
        
        # Setup logging first
        self.setup_logging()
        
        # Initialize the Google Gen AI client
        self.client = genai.Client(api_key=api_key)
        
        # Model configuration
        self.model_config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=None,  # Will be set per request
            max_output_tokens=6000,  # Increased from 4000 to handle larger tables
            temperature=0.1
        )
        
        print(f"Initialized OCR engine with thinking budget: {thinking_budget}")
        print(f"Thinking enabled - Assessment: {enable_thinking_assessment}, OCR: {enable_thinking_ocr}")
        self.logger.info(f"OCR engine initialized with thinking budget: {thinking_budget}")
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
        - Can I actually read specific words or characters in this image?
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
            response_schema=APISchemas.get_combined_assessment_schema(),
            max_output_tokens=4000,
            temperature=0.1
        )
        
        # Add thinking configuration for this specific request if enabled
        if self.enable_thinking_assessment:
            try:
                config.thinking_config = types.ThinkingConfig(
                    thinking_budget=self.thinking_budget
                )
                print(f"Applied thinking config for assessment with budget: {self.thinking_budget}")
            except Exception as e:
                print(f"Could not apply thinking config for assessment: {str(e)}")
        else:
            print("Thinking disabled for assessment phase")
        
        # Convert PIL image to bytes for the API
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
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
            error_msg = f"Error in combined assessment for page {page_num}: {str(e)}"
            print(error_msg)
            self.logger.error(error_msg)
            
            # Log to database if available
            if self.db_logger:
                import traceback
                self.db_logger.log_error(
                    error_type="CombinedAssessmentError",
                    error_message=str(e),
                    stack_trace=traceback.format_exc(),
                    function_name="combined_pre_assessment",
                    severity="high"
                )
            
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
        config.response_schema = APISchemas.get_legibility_schema()
        
        # Convert PIL image to bytes for the API
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
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
    
    def extract_text(self, image: Image.Image, page_num: int) -> OCRResult:
        """Extract text from a legible image using OCR."""
        start_time = time.time()
        
        try:
            prompt = f"""
            Extract ALL text from this scanned page image (Page {page_num}) and format it in proper Markdown syntax. This document may contain text in various languages including English, Sanskrit, Tamil, Hindi, or other Indian languages.
            
            Text Extraction Guidelines:
            1. Capture ALL visible text including headings, paragraphs, footnotes, captions
            2. For Sanskrit text: Preserve all diacritical marks (ā, ī, ū, ṛ, ṝ, ḷ, ḹ, ṃ, ḥ, etc.)
            3. For Tamil text: Preserve all Tamil characters and diacritics accurately
            4. For other Indian language scripts: Maintain authentic character representation
            5. Include any visible verse numbers, section markers, or page headers
            6. Ignore watermarks, page numbers, and purely decorative elements
            7. **2-COLUMN LAYOUT READING ORDER**: If the page has a 2-column layout, read LEFT column first (top to bottom), then RIGHT column (top to bottom). Maintain this left-to-right, top-to-bottom reading sequence in your output.
            8. Maintain the logical reading order based on visual layout
            9. If text is unclear or damaged, do your best to reconstruct based on context
            
            STRICT Markdown Formatting Instructions - Follow These Rules Exactly:
            
            **HEADING HIERARCHY RULES:**
            1. **Major Sections**: Use ## for main chapter/section titles (like "CONTENTS", "CHAPTER I", etc.)
            2. **Chapter Subsections**: Use ### for subsection titles within chapters
            3. **NEVER use multiple # headings for parts of the same title** - combine them into one heading
            
            **LIST FORMATTING RULES - BE CONSISTENT:**
            1. **Table of Contents**: ALWAYS use bullet lists with page numbers
               Format: "* Chapter Title - Page Number"
            2. **Chapter Content Lists**: ALWAYS use bullet points (-) with consistent emphasis
               Format: "- *Topic Title* Page Number" 
            3. **Feature/Topic Lists**: Use bullet points (-) consistently throughout
            4. **NEVER mix numbered and bullet lists within the same section type**
            
            **SPECIFIC FORMATTING PATTERNS:**
            
            For CONTENTS sections:
            ```
            # CONTENTS
            
            ## I SECTION NAME
            * Topic Name - Page Number
            * Another Topic - Page Number
            
            ## II NEXT SECTION  
            * Topic Name - Page Number
            * Another Topic - Page Number
            ```
            
            For chapter topic lists:
            ```
            ## CHAPTER NAME
            - *Topic Title* Page Number
            - *Another Topic* Page Number
            - *Third Topic* Page Number
            ```
            
            **EMPHASIS RULES:**
            - Use *italic* for topic/chapter titles in lists consistently
            - Use **bold** sparingly for very important emphasis only
            - Be consistent within each section type
            
            **SPACING RULES:**
            - Single blank line between different list items
            - Double blank line between major sections
            - No excessive line breaks
            
            **WHAT NOT TO DO - AVOID THESE MISTAKES:**
            - DON'T use multiple # headings for one title (like "# The Essence" then "# of Living Enlightenment")
            - DON'T mix numbered lists and bullet points in the same context
            
            FINAL FORMATTING CHECK:
            Before finalizing your output, review it to ensure:
            1. Consistent heading hierarchy (no multiple # for single titles)
            2. Consistent list formatting within each section type
            3. Consistent emphasis patterns
            4. Proper spacing between sections
            5. All formatting rules above are followed exactly
            
            IMPORTANT: Always provide a confidence score between 0.0 and 1.0 for extraction accuracy.
            Always detect and specify the primary language of the text.
            If no text is visible or extractable, set extracted_text to an empty string but still provide confidence and language fields.
            Return the text in strict, consistent markdown format following ALL the guidelines above.
            """
            
            # Configure for OCR extraction
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=APISchemas.get_ocr_schema(),
                max_output_tokens=6000,  # Increased to handle larger tables
                temperature=0.1
            )
            
            # Add thinking configuration for OCR if enabled
            if self.enable_thinking_ocr:
                try:
                    config.thinking_config = types.ThinkingConfig(
                        thinking_budget=self.thinking_budget
                    )
                    print(f"Applied thinking config for OCR with budget: {self.thinking_budget}")
                except Exception as e:
                    print(f"Could not apply thinking config for OCR: {str(e)}")
            else:
                print("Thinking disabled for OCR phase")
            
            # Convert PIL image to bytes for the API
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            # Add retry logic for API calls
            max_retries = 3
            retry_delay = 2
            
            for attempt in range(max_retries):
                try:
                    self.logger.debug(f"OCR API call attempt {attempt + 1}/{max_retries} for page {page_num}")
                    
                    response = self.client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=[
                            prompt,
                            types.Part.from_bytes(
                                data=img_bytes.getvalue(), 
                                mime_type='image/png'
                            )
                        ],
                        config=config
                    )
                    
                    # If we get here, the API call succeeded
                    self.logger.debug(f"OCR API call succeeded on attempt {attempt + 1} for page {page_num}")
                    break
                    
                except Exception as api_e:
                    self.logger.warning(f"OCR API call attempt {attempt + 1} failed for page {page_num}: {str(api_e)}")
                    
                    if attempt == max_retries - 1:
                        # Last attempt failed, re-raise the exception
                        self.logger.error(f"All {max_retries} OCR API attempts failed for page {page_num}")
                        raise api_e
                    else:
                        # Wait before retrying
                        self.logger.info(f"Retrying OCR API call in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
        
            # Enhanced response validation with detailed logging
            self.logger.debug(f"OCR API response received for page {page_num}")
            self.logger.debug(f"Response type: {type(response)}")
            self.logger.debug(f"Response attributes: {dir(response) if response else 'None'}")
            
            if response is None:
                error_msg = f"API returned None response for page {page_num}"
                print(f"Error: {error_msg}")
                self.logger.error(error_msg)
                raise Exception(error_msg)
            
            # Log response structure for debugging
            self.logger.debug(f"Response has 'text' attribute: {hasattr(response, 'text')}")
            if hasattr(response, 'text'):
                self.logger.debug(f"Response.text is None: {response.text is None}")
                if response.text:
                    self.logger.debug(f"Response.text length: {len(response.text)}")
                    self.logger.debug(f"Response.text preview: {response.text[:200]}...")
            
            self.logger.debug(f"Response has 'candidates': {hasattr(response, 'candidates')}")
            if hasattr(response, 'candidates'):
                self.logger.debug(f"Number of candidates: {len(response.candidates) if response.candidates else 0}")
            
            if not hasattr(response, 'text') or response.text is None:
                # Check if this is a MAX_TOKENS issue and try alternative access
                max_tokens_issue = False
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'finish_reason') and str(candidate.finish_reason) == 'FinishReason.MAX_TOKENS':
                        max_tokens_issue = True
                        self.logger.warning(f"MAX_TOKENS detected for page {page_num}, retrying with increased token limit")
                        print(f"Warning: MAX_TOKENS detected for page {page_num}, retrying with increased token limit")
                
                if max_tokens_issue:
                    # Retry with increased token limit
                    self.logger.info(f"MAX_TOKENS detected for page {page_num}, retrying with increased token limit")
                    try:
                        retry_result = self._retry_with_increased_tokens(image, page_num, prompt, config)
                        return retry_result
                    except Exception as retry_e:
                        self.logger.error(f"Retry with increased tokens also failed for page {page_num}: {str(retry_e)}")
                        # Continue to regular error handling
                else:
                    error_msg = f"API response has no text content for page {page_num}"
                    print(f"Error: {error_msg}")
                    self.logger.error(error_msg)
                    self.logger.error(f"Response object details: {vars(response) if hasattr(response, '__dict__') else 'No __dict__'}")
                    raise Exception(error_msg)
            
            # Extract text content from response with detailed logging
            text_content = None
            
            self.logger.debug(f"Attempting to extract text content for page {page_num}")
            
            if hasattr(response, 'text') and response.text:
                text_content = response.text
                self.logger.debug(f"Text extracted via response.text for page {page_num}")
            elif hasattr(response, 'candidates') and response.candidates:
                self.logger.debug(f"Trying alternative access path via candidates for page {page_num}")
                try:
                    candidate = response.candidates[0]
                    self.logger.debug(f"Candidate type: {type(candidate)}")
                    self.logger.debug(f"Candidate has content: {hasattr(candidate, 'content')}")
                    
                    if hasattr(candidate, 'content') and candidate.content.parts:
                        part = candidate.content.parts[0]
                        self.logger.debug(f"Part type: {type(part)}")
                        self.logger.debug(f"Part has text: {hasattr(part, 'text')}")
                        
                        if hasattr(part, 'text'):
                            text_content = part.text
                            self.logger.debug(f"Text extracted via candidates path for page {page_num}")
                except Exception as alt_e:
                    self.logger.error(f"Error in alternative text extraction for page {page_num}: {str(alt_e)}")
            
            # Use the text content we found
            if not text_content:
                error_msg = f"No text content available for JSON parsing for page {page_num}"
                print(f"Error: {error_msg}")
                self.logger.error(error_msg)
                self.logger.error(f"Failed to extract text content despite response being non-None")
                raise Exception(error_msg)
            
            self.logger.debug(f"Successfully extracted {len(text_content)} characters for page {page_num}")
            
            try:
                result_data = json.loads(text_content)
            except json.JSONDecodeError as je:
                error_msg = f"JSON parsing error in OCR extraction for page {page_num}: {str(je)}"
                print(f"Error: {error_msg}")
                print(f"Problematic response text: {text_content[:500]}...")
                
                # Attempt JSON repair and recovery
                self.logger.warning(f"Attempting JSON repair for page {page_num}")
                recovered_data = self._attempt_json_recovery(text_content, page_num)
                
                if recovered_data:
                    self.logger.info(f"JSON recovery successful for page {page_num}")
                    print(f"JSON recovery successful for page {page_num}")
                    result_data = recovered_data
                else:
                    # If recovery fails, try a simplified retry
                    self.logger.warning(f"JSON recovery failed, attempting simplified extraction for page {page_num}")
                    try:
                        simplified_result = self._retry_with_simplified_prompt(image, page_num)
                        if simplified_result:
                            return simplified_result
                    except Exception as retry_e:
                        self.logger.error(f"Simplified retry also failed for page {page_num}: {str(retry_e)}")
                    
                    # Final fallback - raise the original error
                    raise Exception(error_msg)
            
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
            print(f"Exception args: {e.args}")
            
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
    
    def _retry_with_increased_tokens(self, image: Image.Image, page_num: int, original_prompt: str, original_config: types.GenerateContentConfig) -> OCRResult:
        """Retry OCR extraction with increased token limit when MAX_TOKENS is detected."""
        start_time = time.time()
        
        # Increase token limits progressively
        token_limits = [8000, 12000, 16000]  # Progressive increases
        
        for attempt, max_tokens in enumerate(token_limits, 1):
            self.logger.info(f"Retry attempt {attempt}/{len(token_limits)} with max_output_tokens={max_tokens} for page {page_num}")
            
            # Create new config with increased token limit
            retry_config = types.GenerateContentConfig(
                response_mime_type=original_config.response_mime_type,
                response_schema=original_config.response_schema,
                max_output_tokens=max_tokens,
                temperature=original_config.temperature
            )
            
            # Convert PIL image to bytes for the API
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            try:
                response = self.client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[
                        original_prompt,
                        types.Part.from_bytes(
                            data=img_bytes.getvalue(), 
                            mime_type='image/png'
                        )
                    ],
                    config=retry_config
                )
                
                self.logger.debug(f"Retry attempt {attempt} response received for page {page_num}")
                
                # Check if we got valid text this time
                if response and hasattr(response, 'text') and response.text:
                    try:
                        result_data = json.loads(response.text)
                        processing_time = time.time() - start_time
                        
                        extracted_text = result_data.get('extracted_text', '')
                        confidence = result_data.get('confidence', 0.0)
                        language_detected = result_data.get('language_detected', 'unknown')
                        
                        self.logger.info(f"Retry succeeded on attempt {attempt} for page {page_num}: {len(extracted_text)} characters")
                        print(f"Retry OCR (attempt {attempt}) extracted {len(extracted_text)} characters from page {page_num}")
                        
                        return OCRResult(
                            extracted_text=extracted_text,
                            confidence=confidence,
                            language_detected=language_detected,
                            processing_time=processing_time
                        )
                    except json.JSONDecodeError as je:
                        self.logger.warning(f"Retry attempt {attempt} JSON parsing error for page {page_num}: {str(je)}")
                        if attempt == len(token_limits):  # Last attempt
                            raise Exception(f"All retry attempts failed with JSON parsing errors: {str(je)}")
                        continue
                        
                else:
                    # Check if this is still a MAX_TOKENS issue
                    still_max_tokens = False
                    if hasattr(response, 'candidates') and response.candidates:
                        candidate = response.candidates[0]
                        if hasattr(candidate, 'finish_reason') and str(candidate.finish_reason) == 'FinishReason.MAX_TOKENS':
                            still_max_tokens = True
                            self.logger.warning(f"Retry attempt {attempt} still hit MAX_TOKENS for page {page_num}")
                    
                    if not still_max_tokens:
                        # Different error, not token related
                        raise Exception(f"Retry attempt {attempt}: API response has no text content (not MAX_TOKENS)")
                    
                    if attempt == len(token_limits):  # Last attempt
                        raise Exception(f"All retry attempts with increased tokens failed for page {page_num}")
                        
            except Exception as e:
                self.logger.error(f"Retry attempt {attempt} failed for page {page_num}: {str(e)}")
                if attempt == len(token_limits):  # Last attempt
                    raise e
                continue
        
        # Should not reach here
        raise Exception(f"All retry attempts exhausted for page {page_num}")
    
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
        config.response_schema = APISchemas.get_semantic_schema()
        
        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
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
    
    def _attempt_json_recovery(self, malformed_json: str, page_num: int) -> dict:
        """
        Attempt to recover usable data from malformed JSON responses.
        
        Args:
            malformed_json: The malformed JSON string from Gemini
            page_num: Page number for logging
            
        Returns:
            Recovered dictionary or None if recovery fails
        """
        import re
        
        self.logger.debug(f"Attempting JSON recovery for page {page_num}")
        
        # Strategy 1: Try to repair common JSON issues
        recovery_strategies = [
            self._repair_truncated_json,
            self._repair_unescaped_quotes,
            self._extract_content_with_regex,
            self._extract_partial_content
        ]
        
        for strategy_num, strategy in enumerate(recovery_strategies, 1):
            try:
                self.logger.debug(f"Trying recovery strategy {strategy_num} for page {page_num}")
                result = strategy(malformed_json, page_num)
                if result:
                    self.logger.info(f"Recovery strategy {strategy_num} succeeded for page {page_num}")
                    return result
            except Exception as e:
                self.logger.debug(f"Recovery strategy {strategy_num} failed for page {page_num}: {str(e)}")
                continue
        
        self.logger.warning(f"All JSON recovery strategies failed for page {page_num}")
        return None
    
    def _repair_truncated_json(self, json_str: str, page_num: int) -> dict:
        """Attempt to repair JSON that was truncated mid-string."""
        import json
        
        # Check if JSON ends abruptly in a string
        if json_str.rstrip().endswith('...') or not json_str.rstrip().endswith('}'):
            self.logger.debug(f"Detected truncated JSON for page {page_num}")
            
            # Find the last complete field before truncation
            lines = json_str.split('\n')
            repaired_lines = []
            in_string = False
            
            for line in lines:
                # If we hit an incomplete string, try to close it
                if '"extracted_text":' in line and not line.rstrip().endswith('"'):
                    # Find the opening quote and close the string
                    if '": "' in line:
                        parts = line.split('": "', 1)
                        if len(parts) == 2:
                            # Close the string and add remaining required fields
                            repaired_lines.append(parts[0] + '": "' + parts[1].rstrip() + '"')
                            break
                else:
                    repaired_lines.append(line)
            
            # Add missing required fields if needed
            repaired_json = '\n'.join(repaired_lines)
            if not repaired_json.rstrip().endswith('}'):
                repaired_json = repaired_json.rstrip() + ',\n  "confidence": 0.7,\n  "language_detected": "English"\n}'
            
            try:
                return json.loads(repaired_json)
            except json.JSONDecodeError:
                pass
        
        return None
    
    def _repair_unescaped_quotes(self, json_str: str, page_num: int) -> dict:
        """Attempt to repair JSON with unescaped quotes in strings."""
        import json
        import re
        
        # Replace unescaped quotes within string values
        # This is a simple heuristic - look for quotes between ": " and ",
        pattern = r'(": ")([^"]*)"([^"]*")([^"]*")([^",}]*)(["|}])'
        
        def escape_quotes(match):
            prefix = match.group(1)
            content = (match.group(2) + '\\"' + match.group(3) + '\\"' + match.group(4) + '\\"' + match.group(5))
            suffix = match.group(6)
            return prefix + content + suffix
        
        repaired = re.sub(pattern, escape_quotes, json_str)
        
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            return None
    
    def _extract_content_with_regex(self, json_str: str, page_num: int) -> dict:
        """Extract content using regex when JSON parsing fails completely."""
        import re
        
        # Try to extract the main content even if JSON is malformed
        extracted_text = ""
        confidence = 0.5  # Default for recovered content
        language = "English"  # Default
        
        # Extract text content from extracted_text field
        text_match = re.search(r'"extracted_text":\s*"([^"]*(?:\\"[^"]*)*)', json_str, re.DOTALL)
        if text_match:
            extracted_text = text_match.group(1).replace('\\"', '"')
        
        # Extract confidence if available
        conf_match = re.search(r'"confidence":\s*([0-9.]+)', json_str)
        if conf_match:
            confidence = float(conf_match.group(1))
        
        # Extract language if available
        lang_match = re.search(r'"language_detected":\s*"([^"]+)"', json_str)
        if lang_match:
            language = lang_match.group(1)
        
        if extracted_text:
            self.logger.info(f"Extracted content via regex for page {page_num}: {len(extracted_text)} characters")
            return {
                "extracted_text": extracted_text,
                "confidence": confidence,
                "language_detected": language
            }
        
        return None
    
    def _extract_partial_content(self, json_str: str, page_num: int) -> dict:
        """Extract whatever content is available, even if incomplete."""
        # As a last resort, try to extract any visible text content
        # Look for content that appears to be extracted text
        
        lines = json_str.split('\n')
        content_lines = []
        capturing = False
        
        for line in lines:
            if '"extracted_text":' in line:
                capturing = True
                # Extract content from this line if any
                if '": "' in line:
                    content_start = line.split('": "', 1)[1]
                    content_lines.append(content_start.rstrip('"'))
                continue
            
            if capturing:
                # Stop if we hit another JSON field
                if line.strip().startswith('"') and '":' in line:
                    break
                content_lines.append(line)
        
        if content_lines:
            extracted_content = '\n'.join(content_lines).strip()
            if extracted_content:
                self.logger.info(f"Extracted partial content for page {page_num}: {len(extracted_content)} characters")
                return {
                    "extracted_text": extracted_content,
                    "confidence": 0.4,  # Lower confidence for partial extraction
                    "language_detected": "English"
                }
        
        return None
    
    def _retry_with_simplified_prompt(self, image: Image.Image, page_num: int) -> OCRResult:
        """
        Retry OCR with a simplified prompt that's less likely to cause JSON issues.
        
        Args:
            image: PIL Image to process
            page_num: Page number for logging
            
        Returns:
            OCRResult or None if retry fails
        """
        start_time = time.time()
        
        # Simplified prompt that's less likely to produce large responses
        simplified_prompt = f"""
        Extract the main text content from this image (Page {page_num}). 
        
        Focus on:
        1. Main headings and titles
        2. Primary text content
        3. Key information
        
        Ignore:
        - Large tables (just mention "Table with X columns")
        - Detailed formatting
        - Minor text elements
        
        Keep the response concise but capture the essential content.
        """
        
        try:
            # Use smaller token limit for simplified extraction
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=APISchemas.get_ocr_schema(),
                max_output_tokens=2000,  # Smaller limit
                temperature=0.1
            )
            
            # Convert PIL image to bytes for the API
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[
                    simplified_prompt,
                    types.Part.from_bytes(
                        data=img_bytes.getvalue(), 
                        mime_type='image/png'
                    )
                ],
                config=config
            )
            
            if response and response.text:
                try:
                    result_data = json.loads(response.text)
                    processing_time = time.time() - start_time
                    
                    self.logger.info(f"Simplified retry successful for page {page_num}")
                    print(f"Simplified retry successful for page {page_num}")
                    
                    return OCRResult(
                        extracted_text=result_data.get('extracted_text', ''),
                        confidence=result_data.get('confidence', 0.5),
                        language_detected=result_data.get('language_detected', 'English'),
                        processing_time=processing_time
                    )
                except json.JSONDecodeError:
                    self.logger.warning(f"Simplified retry also produced malformed JSON for page {page_num}")
            
        except Exception as e:
            self.logger.error(f"Simplified retry failed for page {page_num}: {str(e)}")
        
        return None
    
