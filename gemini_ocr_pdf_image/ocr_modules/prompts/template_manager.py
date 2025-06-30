"""
Prompt template management for OCR operations.

This module centralizes all prompt templates used in OCR processing,
enabling easy customization, A/B testing, and maintenance.
"""

from typing import Dict, Any
from string import Template


class PromptTemplateManager:
    """
    Manages prompt templates for different OCR operations.
    
    Provides centralized storage and rendering of prompts with
    parameter substitution and template customization support.
    """
    
    def __init__(self):
        """Initialize the template manager with default templates."""
        self._templates = {
            'combined_assessment': self._get_combined_assessment_template(),
            'ocr_extraction': self._get_ocr_extraction_template(),
            'legibility_assessment': self._get_legibility_assessment_template(),
            'semantic_validation': self._get_semantic_validation_template(),
            'simplified_extraction': self._get_simplified_extraction_template()
        }
    
    def get_prompt(self, template_name: str, **kwargs) -> str:
        """
        Get a rendered prompt template with parameter substitution.
        
        Args:
            template_name: Name of the template to use
            **kwargs: Parameters to substitute in the template
            
        Returns:
            Rendered prompt string
            
        Raises:
            KeyError: If template_name is not found
        """
        if template_name not in self._templates:
            available = ', '.join(self._templates.keys())
            raise KeyError(f"Template '{template_name}' not found. Available: {available}")
        
        template = Template(self._templates[template_name])
        return template.safe_substitute(**kwargs)
    
    def list_templates(self) -> list:
        """Get a list of available template names."""
        return list(self._templates.keys())
    
    def update_template(self, template_name: str, template_content: str) -> None:
        """
        Update or add a template.
        
        Args:
            template_name: Name of the template
            template_content: Template content with $variable placeholders
        """
        self._templates[template_name] = template_content
    
    def _get_combined_assessment_template(self) -> str:
        """Get the combined pre-assessment prompt template."""
        return """
        THINK CAREFULLY about this scanned document image (Page $page_num) to determine if OCR processing will produce meaningful, usable text.
        
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
        - Legibility score: How clearly can the text be seen? ($legibility_threshold minimum needed)
        - Semantic quality: Would OCR produce meaningful, usable text? ($semantic_threshold minimum needed)
        
        **DECISION CRITERIA - BE STRICT:**
        Set should_process=true ONLY if you are confident that:
        1. The text is visually clear enough for accurate OCR (legibility >= $legibility_threshold)
        2. OCR would produce meaningful, coherent text rather than gibberish (semantic >= $semantic_threshold)
        
        **IMPORTANT:** If you see signs that OCR would produce fragmented words, nonsensical combinations, or mostly gibberish, set a LOW semantic quality score (< $semantic_threshold) to prevent wasting processing time.
        
        Provide a sample of the most clearly visible text you can actually read from the image.
        """
    
    def _get_ocr_extraction_template(self) -> str:
        """Get the OCR text extraction prompt template."""
        return """
            Extract ALL text from this scanned page image (Page $page_num) and format it in proper Markdown syntax. This document may contain text in various languages including English, Sanskrit, Tamil, Hindi, or other Indian languages.
            
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
    
    def _get_legibility_assessment_template(self) -> str:
        """Get the legibility assessment prompt template."""
        return """
        Analyze this scanned page image (Page $page_num) to determine if the text is legible enough for OCR processing.
        
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
    
    def _get_semantic_validation_template(self) -> str:
        """Get the semantic validation prompt template."""
        return """
        Analyze this extracted OCR text to determine if it's semantically meaningful and coherent in $language.
        
        Text to analyze:
        ```
        $text
        ```
        
        Evaluation criteria:
        
        1. **Language Consistency**: 
           - Is the text consistently in $language?
           - Are there random character combinations that don't form words?
           - Is there mixing of scripts or languages inappropriately?
        
        2. **Word Formation**: 
           - Do the words follow proper formation rules for $language?
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
    
    def _get_simplified_extraction_template(self) -> str:
        """Get the simplified OCR extraction prompt template for error recovery."""
        return """
        Extract the main text content from this image (Page $page_num). 
        
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