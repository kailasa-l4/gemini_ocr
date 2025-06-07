"""
JSON schemas for Gemini API responses.
"""


class APISchemas:
    """Container for all API response schemas."""
    
    @staticmethod
    def get_legibility_schema():
        """Schema for legibility assessment responses."""
        return {
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
    
    @staticmethod
    def get_ocr_schema():
        """Schema for OCR extraction responses."""
        return {
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
    
    @staticmethod
    def get_semantic_schema():
        """Schema for semantic validation responses."""
        return {
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
    
    @staticmethod
    def get_combined_assessment_schema():
        """Schema for combined pre-assessment responses."""
        return {
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