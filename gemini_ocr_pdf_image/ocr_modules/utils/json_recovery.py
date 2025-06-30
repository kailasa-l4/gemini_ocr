"""
JSON recovery utilities for handling malformed API responses.

This module provides strategies for recovering usable data from
malformed JSON responses, particularly from large table OCR operations.
"""

import json
import re
import logging
from typing import Optional, Dict, Any


class JSONRecoveryService:
    """
    Service for recovering data from malformed JSON responses.
    
    Provides multiple recovery strategies to handle common JSON parsing
    errors from large content extraction operations.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the JSON recovery service.
        
        Args:
            logger: Optional logger instance for debugging
        """
        self.logger = logger or logging.getLogger('JSONRecoveryService')
    
    def recover_json(self, malformed_json: str, context: str = "operation") -> Optional[Dict[str, Any]]:
        """
        Attempt to recover usable data from malformed JSON.
        
        Args:
            malformed_json: The malformed JSON string
            context: Context description for logging
            
        Returns:
            Recovered dictionary or None if all recovery fails
        """
        self.logger.debug(f"Attempting JSON recovery for {context}")
        
        # Multiple recovery strategies in order of preference
        recovery_strategies = [
            self._repair_truncated_json,
            self._repair_unescaped_quotes,
            self._extract_content_with_regex,
            self._extract_partial_content
        ]
        
        for strategy_num, strategy in enumerate(recovery_strategies, 1):
            try:
                self.logger.debug(f"Trying recovery strategy {strategy_num} for {context}")
                result = strategy(malformed_json, context)
                if result:
                    self.logger.info(f"Recovery strategy {strategy_num} succeeded for {context}")
                    return result
            except Exception as e:
                self.logger.debug(f"Recovery strategy {strategy_num} failed for {context}: {str(e)}")
                continue
        
        self.logger.warning(f"All JSON recovery strategies failed for {context}")
        return None
    
    def _repair_truncated_json(self, json_str: str, context: str) -> Optional[Dict[str, Any]]:
        """
        Attempt to repair JSON that was truncated mid-string.
        
        Args:
            json_str: Malformed JSON string
            context: Context for logging
            
        Returns:
            Repaired dictionary or None
        """
        # Check if JSON ends abruptly in a string
        if json_str.rstrip().endswith('...') or not json_str.rstrip().endswith('}'):
            self.logger.debug(f"Detected truncated JSON for {context}")
            
            # Find the last complete field before truncation
            lines = json_str.split('\n')
            repaired_lines = []
            
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
    
    def _repair_unescaped_quotes(self, json_str: str, context: str) -> Optional[Dict[str, Any]]:
        """
        Attempt to repair JSON with unescaped quotes in strings.
        
        Args:
            json_str: Malformed JSON string
            context: Context for logging
            
        Returns:
            Repaired dictionary or None
        """
        # Replace unescaped quotes within string values
        # This is a simple heuristic - look for quotes between ": " and ",
        pattern = r'(": ")([^"]*)"([^"]*")([^"]*")([^",}]*)(["|}])'
        
        def escape_quotes(match):
            prefix = match.group(1)
            content = (match.group(2) + '\\"' + match.group(3) + '\\"' + 
                      match.group(4) + '\\"' + match.group(5))
            suffix = match.group(6)
            return prefix + content + suffix
        
        repaired = re.sub(pattern, escape_quotes, json_str)
        
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            return None
    
    def _extract_content_with_regex(self, json_str: str, context: str) -> Optional[Dict[str, Any]]:
        """
        Extract content using regex when JSON parsing fails completely.
        
        Args:
            json_str: Malformed JSON string
            context: Context for logging
            
        Returns:
            Extracted dictionary or None
        """
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
            self.logger.info(f"Extracted content via regex for {context}: {len(extracted_text)} characters")
            return {
                "extracted_text": extracted_text,
                "confidence": confidence,
                "language_detected": language
            }
        
        return None
    
    def _extract_partial_content(self, json_str: str, context: str) -> Optional[Dict[str, Any]]:
        """
        Extract whatever content is available, even if incomplete.
        
        Args:
            json_str: Malformed JSON string
            context: Context for logging
            
        Returns:
            Partial content dictionary or None
        """
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
                self.logger.info(f"Extracted partial content for {context}: {len(extracted_content)} characters")
                return {
                    "extracted_text": extracted_content,
                    "confidence": 0.4,  # Lower confidence for partial extraction
                    "language_detected": "English"
                }
        
        return None
    
    def validate_and_recover(self, response_text: str, context: str = "operation") -> Dict[str, Any]:
        """
        Validate JSON response and attempt recovery if needed.
        
        Args:
            response_text: Raw response text from API
            context: Context description for logging
            
        Returns:
            Parsed JSON dictionary
            
        Raises:
            Exception: If both parsing and recovery fail
        """
        try:
            # Try normal JSON parsing first
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            self.logger.warning(f"JSON parsing failed for {context}: {str(e)}")
            
            # Attempt recovery
            recovered_data = self.recover_json(response_text, context)
            
            if recovered_data:
                self.logger.info(f"JSON recovery successful for {context}")
                return recovered_data
            else:
                # Recovery failed, raise original error
                raise Exception(f"JSON parsing failed and recovery unsuccessful for {context}: {str(e)}")