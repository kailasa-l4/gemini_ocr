#!/usr/bin/env python3
"""
Test script for database integration
"""

import os
from dotenv import load_dotenv
from ocr_modules.db_logger import DatabaseLogger

def test_database_connection():
    """Test database connection and basic operations."""
    load_dotenv()
    
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        print("Error: DATABASE_URL not found in environment")
        return False
    
    print(f"Testing database connection to: {database_url}")
    
    try:
        # Test connection
        db_logger = DatabaseLogger(
            database_url=database_url,
            enabled=True,
            connection_timeout=30,
            retry_attempts=3
        )
        print("✓ Database connection successful")
        
        # Test session creation
        session_id = db_logger.start_session(
            input_path="test_file.pdf",
            input_type="pdf",
            output_path="./test_output",
            configuration={
                "test": True,
                "legibility_threshold": 0.5
            }
        )
        print(f"✓ Session created: {session_id}")
        
        # Test processing log
        db_logger.log_processing_complete(
            session_id=session_id,
            file_path="test_file.pdf",
            page_number=1,
            status="completed",
            legibility_score=0.95,
            semantic_score=0.90,
            ocr_confidence=0.98,
            processing_time=5.2,
            text_clarity="excellent",
            image_quality="good",
            ocr_prediction="excellent",
            semantic_prediction="meaningful_text",
            visible_text_sample="This is a test page",
            language_detected="English",
            issues_found="none"
        )
        print("✓ Processing log created")
        
        # Test error logging
        db_logger.log_error(
            error_type="TestError",
            error_message="This is a test error",
            stack_trace="test stack trace",
            file_path="test_file.pdf",
            function_name="test_function",
            severity="low",
            session_id=session_id
        )
        print("✓ Error log created")
        
        # Test session completion
        db_logger.update_session(session_id, total_files=1, completed_files=1, failed_files=0)
        db_logger.end_session(session_id, 'completed')
        print("✓ Session completed")
        
        # Test session stats
        stats = db_logger.get_session_stats(session_id)
        print(f"✓ Session stats retrieved: {len(stats)} records")
        
        # Close connections
        db_logger.close()
        print("✓ Database connections closed")
        
        print("\n🎉 All database tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Database test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    test_database_connection()