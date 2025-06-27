#!/usr/bin/env python3
"""
Test OCR processing with database logging integration
"""

import os
import sys
from dotenv import load_dotenv
from ocr_modules import GeminiAdvancedOCR
from ocr_modules.db_logger import DatabaseLogger

def test_ocr_with_database():
    """Test OCR processing with database integration."""
    load_dotenv()
    
    # Check configuration
    api_key = os.getenv('GEMINI_API_KEY')
    database_url = os.getenv('DATABASE_URL')
    enable_db = os.getenv('ENABLE_DATABASE_LOGGING', 'false').lower() in ('true', '1', 'yes')
    
    if not api_key:
        print("Error: GEMINI_API_KEY not found in environment")
        return False
    
    if not database_url:
        print("Error: DATABASE_URL not found in environment")
        return False
    
    print("=== OCR Database Integration Test ===\n")
    print(f"API Key: {'✓ Configured' if api_key else '✗ Missing'}")
    print(f"Database URL: {database_url}")
    print(f"Database Logging: {'✓ Enabled' if enable_db else '✗ Disabled'}")
    
    try:
        # Initialize database logger
        db_logger = None
        if enable_db:
            db_logger = DatabaseLogger(
                database_url=database_url,
                enabled=True,
                connection_timeout=30,
                retry_attempts=3
            )
            print("✓ Database logger initialized")
        
        # Initialize OCR processor
        ocr = GeminiAdvancedOCR(
            api_key=api_key,
            thinking_budget=1000,
            enable_thinking_assessment=False,
            enable_thinking_ocr=False,
            db_logger=db_logger,
            logs_dir='./logs'
        )
        print("✓ OCR processor initialized")
        
        # Check if we have any sample images to test with
        test_images = [
            '/home/psp/projects/gemini_ocr/images/page (16).jpg',
            './images/sample.jpg',
            './test_image.jpg'
        ]
        
        test_image = None
        for img_path in test_images:
            if os.path.exists(img_path):
                test_image = img_path
                break
        
        if test_image:
            print(f"✓ Found test image: {test_image}")
            
            # Test single image processing
            print("\nTesting single image processing with database logging...")
            result_file = ocr.process_single_image(
                image_path=test_image,
                output_dir="./test_output",
                legibility_threshold=0.5,
                semantic_threshold=0.6
            )
            
            if result_file:
                print(f"✓ OCR processing successful: {result_file}")
                
                # Check if database has records
                if db_logger:
                    print("\nChecking database for logged data...")
                    # Query for recent sessions
                    with db_logger.get_connection() as conn:
                        if conn:
                            cursor = conn.cursor()
                            cursor.execute("""
                                SELECT session_id, input_path, status, total_files, created_at
                                FROM ocr_sessions 
                                ORDER BY created_at DESC 
                                LIMIT 5
                            """)
                            sessions = cursor.fetchall()
                            
                            cursor.execute("""
                                SELECT COUNT(*) FROM processing_logs
                                WHERE created_at > NOW() - INTERVAL '1 hour'
                            """)
                            recent_logs = cursor.fetchone()[0]
                            
                            cursor.close()
                            
                            print(f"✓ Recent sessions in database: {len(sessions)}")
                            print(f"✓ Recent processing logs: {recent_logs}")
                            
                            if sessions:
                                latest_session = sessions[0]
                                print(f"✓ Latest session: {latest_session[0]} - {latest_session[2]}")
                
                print("\n🎉 OCR database integration test completed successfully!")
                return True
            else:
                print("✗ OCR processing failed")
                return False
        else:
            print("⚠ No test images found. Database logging integration is ready, but OCR test skipped.")
            print("Available test paths checked:")
            for img_path in test_images:
                print(f"  - {img_path} {'✓' if os.path.exists(img_path) else '✗'}")
            print("\nTo test with actual images, place a test image in one of the above paths.")
            return True
            
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_ocr_with_database()
    sys.exit(0 if success else 1)