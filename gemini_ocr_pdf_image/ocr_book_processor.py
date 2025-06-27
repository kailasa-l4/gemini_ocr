"""
Enhanced OCR Script with Gemini 2.5 Flash and Combined Pre-Assessment

Features:
- Supports PDF files, single image files, and directories of image files
- Optimized combined pre-assessment: Visual legibility + Semantic quality prediction in one API call
- Structured outputs using JSON schemas
- Progress tracking with CSV files
- Resume functionality for interrupted processing
- Markdown output for all extracted text
- Recursive directory scanning for images
- Configurable thresholds for both visual and semantic validation
- Reduced API consumption (2 calls per image instead of 3)

Supported image formats: JPG, JPEG, PNG, BMP, TIFF, TIF, WEBP

Usage:
    # Process a PDF file
    python ocr_book_processor.py --input-file book.pdf --output-dir ./output --api-key YOUR_KEY
    
    # Process a single image file with custom thresholds
    python ocr_book_processor.py --input-file old_document.png --output-dir ./output --api-key YOUR_KEY \
        --legibility-threshold 0.6 --semantic-threshold 0.7
    
    # Process a directory of images
    python ocr_book_processor.py --input-dir ./images --output-dir ./output --api-key YOUR_KEY
"""

import os
import argparse
from dotenv import load_dotenv
from ocr_modules import GeminiAdvancedOCR, is_pdf_file, is_image_file, find_all_supported_files


def load_env_config():
    """Load configuration from .env file."""
    load_dotenv()
    
    config = {}
    
    # Load all possible environment variables
    if os.getenv('GEMINI_API_KEY'):
        config['api_key'] = os.getenv('GEMINI_API_KEY')
    if os.getenv('INPUT_FILE'):
        config['input_file'] = os.getenv('INPUT_FILE')
    if os.getenv('INPUT_DIR'):
        config['input_dir'] = os.getenv('INPUT_DIR')
    if os.getenv('OUTPUT_DIR'):
        config['output_dir'] = os.getenv('OUTPUT_DIR')
    if os.getenv('START_PAGE'):
        config['start_page'] = int(os.getenv('START_PAGE'))
    if os.getenv('END_PAGE'):
        config['end_page'] = int(os.getenv('END_PAGE'))
    if os.getenv('DPI'):
        config['dpi'] = int(os.getenv('DPI'))
    if os.getenv('LEGIBILITY_THRESHOLD'):
        config['legibility_threshold'] = float(os.getenv('LEGIBILITY_THRESHOLD'))
    if os.getenv('SEMANTIC_THRESHOLD'):
        config['semantic_threshold'] = float(os.getenv('SEMANTIC_THRESHOLD'))
    if os.getenv('THINKING_BUDGET'):
        config['thinking_budget'] = int(os.getenv('THINKING_BUDGET'))
    if os.getenv('ENABLE_THINKING_ASSESSMENT'):
        config['enable_thinking_assessment'] = os.getenv('ENABLE_THINKING_ASSESSMENT').lower() in ('true', '1', 'yes')
    if os.getenv('ENABLE_THINKING_OCR'):
        config['enable_thinking_ocr'] = os.getenv('ENABLE_THINKING_OCR').lower() in ('true', '1', 'yes')
    
    # Database logging configuration
    if os.getenv('ENABLE_DATABASE_LOGGING'):
        config['enable_database_logging'] = os.getenv('ENABLE_DATABASE_LOGGING').lower() in ('true', '1', 'yes')
    if os.getenv('DATABASE_URL'):
        config['database_url'] = os.getenv('DATABASE_URL')
    if os.getenv('DATABASE_CONNECTION_TIMEOUT'):
        config['database_connection_timeout'] = int(os.getenv('DATABASE_CONNECTION_TIMEOUT'))
    if os.getenv('DATABASE_RETRY_ATTEMPTS'):
        config['database_retry_attempts'] = int(os.getenv('DATABASE_RETRY_ATTEMPTS'))
    
    # Logging directory configuration
    if os.getenv('LOGS_DIR'):
        config['logs_dir'] = os.getenv('LOGS_DIR')
    if os.getenv('CSV_LOGS_DIR'):
        config['csv_logs_dir'] = os.getenv('CSV_LOGS_DIR')
    
    return config


def main():
    # Load environment configuration first
    env_config = load_env_config()
    
    parser = argparse.ArgumentParser(
        description="Enhanced OCR with Gemini 2.5 Flash and legibility detection",
        epilog="Configuration can be loaded from .env file. CLI arguments override .env values."
    )
    
    # Input options - PDF file, image file, or image directory
    input_group = parser.add_mutually_exclusive_group(required=not (env_config.get('input_file') or env_config.get('input_dir')))
    input_group.add_argument("--input-file", 
                           default=env_config.get('input_file'),
                           help="PDF file or single image file to process")
    input_group.add_argument("--input-dir", 
                           default=env_config.get('input_dir'),
                           help="Directory containing image files")
    
    parser.add_argument("--output-dir", 
                       default=env_config.get('output_dir'),
                       required=not env_config.get('output_dir'),
                       help="Output directory")
    parser.add_argument("--api-key", 
                       default=env_config.get('api_key'),
                       required=not env_config.get('api_key'),
                       help="Gemini API key")
    
    # PDF-specific options
    parser.add_argument("--start-page", type=int, 
                       default=env_config.get('start_page', 1), 
                       help="Start page for PDF (1-based)")
    parser.add_argument("--end-page", type=int, 
                       default=env_config.get('end_page'),
                       help="End page for PDF (inclusive)")
    parser.add_argument("--dpi", type=int, 
                       default=env_config.get('dpi', 300), 
                       help="DPI for PDF rendering")
    
    # Common options
    parser.add_argument("--legibility-threshold", type=float, 
                       default=env_config.get('legibility_threshold', 0.5), 
                       help="Minimum legibility score (0-1)")
    parser.add_argument("--semantic-threshold", type=float, 
                       default=env_config.get('semantic_threshold', 0.6), 
                       help="Minimum semantic meaningfulness score (0-1)")
    parser.add_argument("--thinking-budget", type=int, 
                       default=env_config.get('thinking_budget', 2000), 
                       help="Thinking budget for Gemini")
    parser.add_argument("--enable-thinking-assessment", action="store_true", 
                       default=env_config.get('enable_thinking_assessment', True),
                       help="Enable thinking for assessment phase")
    parser.add_argument("--enable-thinking-ocr", action="store_true", 
                       default=env_config.get('enable_thinking_ocr', False),
                       help="Enable thinking for OCR extraction phase")
    
    # Database logging options
    parser.add_argument("--enable-database-logging", action="store_true", 
                       default=env_config.get('enable_database_logging', False),
                       help="Enable database logging")
    parser.add_argument("--disable-database-logging", action="store_true", 
                       help="Disable database logging (override .env)")
    parser.add_argument("--database-url", 
                       default=env_config.get('database_url'),
                       help="PostgreSQL database URL")
    
    args = parser.parse_args()
    
    # Handle database logging flags
    database_logging_enabled = args.enable_database_logging
    if args.disable_database_logging:
        database_logging_enabled = False
    
    # Validate database connection if enabled
    db_logger = None
    if database_logging_enabled:
        if not args.database_url:
            print("Error: Database logging is enabled but no database URL provided.")
            print("Please provide --database-url or set DATABASE_URL in .env file.")
            return
        
        try:
            from ocr_modules.db_logger import DatabaseLogger
            db_logger = DatabaseLogger(
                database_url=args.database_url,
                enabled=True,
                connection_timeout=env_config.get('database_connection_timeout', 30),
                retry_attempts=env_config.get('database_retry_attempts', 3)
            )
            print("✓ Database connection validated successfully")
        except Exception as e:
            print(f"✗ Database connection failed: {e}")
            print("OCR processing cannot continue with database logging enabled.")
            print("Either fix the database connection or disable database logging.")
            return
    else:
        print("Database logging is disabled")
    
    # Validate inputs
    if args.input_file and not os.path.exists(args.input_file):
        print(f"Error: File not found: {args.input_file}")
        return
    
    if args.input_dir and not os.path.exists(args.input_dir):
        print(f"Error: Directory not found: {args.input_dir}")
        return
    
    # Create OCR processor
    ocr = GeminiAdvancedOCR(
        args.api_key, 
        thinking_budget=args.thinking_budget,
        enable_thinking_assessment=args.enable_thinking_assessment,
        enable_thinking_ocr=args.enable_thinking_ocr,
        db_logger=db_logger,
        logs_dir=env_config.get('logs_dir', './logs')
    )
    
    # Process based on input type
    if args.input_file:
        if is_pdf_file(args.input_file):
            # Process as PDF
            print(f"\n{'='*80}\nProcessing PDF: {args.input_file}\n{'='*80}")
            result_file = ocr.process_pdf(
                args.input_file,
                args.output_dir,
                args.start_page,
                args.end_page,
                args.dpi,
                args.legibility_threshold,
                args.semantic_threshold
            )
            print(f"\nPDF processing complete! Final file: {result_file}")
            
        elif is_image_file(args.input_file):
            # Process as single image
            print(f"\n{'='*80}\nProcessing single image: {args.input_file}\n{'='*80}")
            result_file = ocr.process_single_image(
                args.input_file,
                args.output_dir,
                args.legibility_threshold,
                args.semantic_threshold
            )
            print(f"\nImage processing complete! Final file: {result_file}")
            
        else:
            print(f"Error: Unsupported file format: {args.input_file}")
            print("Supported formats: PDF, JPG, JPEG, PNG, BMP, TIFF, TIF, WEBP")
            return
        
    elif args.input_dir:
        print(f"\n{'='*80}\nProcessing directory: {args.input_dir}\n{'='*80}")
        
        # Find all supported files (PDFs and images)
        pdf_files, image_files = find_all_supported_files(args.input_dir)
        
        if not pdf_files and not image_files:
            print("No supported files found!")
            print("Supported formats: PDF, JPG, JPEG, PNG, BMP, TIFF, TIF, WEBP")
            return
        
        print(f"Found {len(pdf_files)} PDF files and {len(image_files)} image files")
        
        result_files = []
        
        # Process each PDF individually
        for pdf_file in pdf_files:
            print(f"\n{'='*60}\nProcessing PDF: {pdf_file}\n{'='*60}")
            result_file = ocr.process_pdf(
                pdf_file,
                args.output_dir,
                args.start_page,
                args.end_page,
                args.dpi,
                args.legibility_threshold,
                args.semantic_threshold
            )
            if result_file:
                result_files.append(result_file)
                print(f"PDF processing complete! Final file: {result_file}")
        
        # Process images as a batch (existing behavior)
        if image_files:
            print(f"\n{'='*60}\nProcessing {len(image_files)} images as batch\n{'='*60}")
            result_file = ocr.process_images(
                args.input_dir,
                args.output_dir,
                args.legibility_threshold,
                args.semantic_threshold
            )
            if result_file:
                result_files.append(result_file)
                print(f"Image batch processing complete! Final file: {result_file}")
        
        print(f"\nDirectory processing complete! Created {len(result_files)} output files.")


if __name__ == "__main__":
    main()