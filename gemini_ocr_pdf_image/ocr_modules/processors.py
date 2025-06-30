"""
Document processors for PDF files, single images, and image directories.
"""

import time
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional
from tqdm import tqdm
from PIL import Image
import fitz  # PyMuPDF

from .models import PageProgress, ImageProgress
from .progress_manager import ProgressManager
from .ocr_engine import GeminiOCREngine
from .document import PDFHandler, ImageHandler, ContentAggregator
from .utils import find_image_files, get_safe_filename


class PDFProcessor:
    """Handles PDF document processing."""
    
    def __init__(self, ocr_engine: GeminiOCREngine, db_logger=None, logs_dir: str = './logs'):
        self.ocr_engine = ocr_engine
        self.db_logger = db_logger
        self.logs_dir = logs_dir
        self.progress_manager = ProgressManager(db_logger=db_logger, logs_dir=logs_dir)
        self.pdf_handler = PDFHandler()
        self.content_aggregator = ContentAggregator()
    
    def process_pdf(self, pdf_path: str, output_dir: str, start_page: int = 1, 
                    end_page: Optional[int] = None, dpi: int = 300, 
                    legibility_threshold: float = 0.5, semantic_threshold: float = 0.6,
                    file_progress: Optional[tuple] = None) -> str:
        """
        Process a PDF file with enhanced OCR and legibility detection.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save output
            start_page: First page to process (1-based)
            end_page: Last page to process (inclusive)
            dpi: DPI for rendering PDF pages
            legibility_threshold: Minimum legibility score to attempt OCR
            
        Returns:
            Path to the final processed markdown file
        """
        # Start database session if available
        session_id = None
        if self.db_logger:
            session_id = self.db_logger.start_session(
                input_path=pdf_path,
                input_type='pdf',
                output_path=output_dir,
                configuration={
                    'start_page': start_page,
                    'end_page': end_page,
                    'dpi': dpi,
                    'legibility_threshold': legibility_threshold,
                    'semantic_threshold': semantic_threshold
                }
            )
        
        # Setup output paths
        book_name = Path(pdf_path).stem
        book_output_dir = Path(output_dir) / book_name
        book_output_dir.mkdir(parents=True, exist_ok=True)
        
        progress_file = book_output_dir / f"{book_name}_progress.csv"
        final_output_file = book_output_dir / f"{book_name}_processed.md"
        
        # Open PDF to get total pages
        print(f"Opening PDF: {pdf_path}")
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        print(f"PDF has {total_pages} pages")
        
        # Adjust page range
        if start_page < 1:
            start_page = 1
        if end_page is None or end_page > total_pages:
            end_page = total_pages
            
        pages_to_process = range(start_page - 1, end_page)  # 0-based index
        num_pages = len(pages_to_process)
        
        # Use fast resume method to get completion status
        completion_status = self.progress_manager.get_completion_status_fast(
            str(progress_file), num_pages
        )
        
        completed_pages = completion_status['completed_count']
        remaining_pages = completion_status['remaining_count']
        completed_page_set = completion_status['completed_pages']
        
        print(f"Processing pages {start_page} to {end_page} ({num_pages} pages)")
        print(f"Progress: {completed_pages}/{num_pages} completed ({remaining_pages} remaining)")
        print(f"Completion: {(completed_pages/num_pages)*100:.1f}%")
        print(f"Legibility threshold: {legibility_threshold}")
        print(f"Semantic threshold: {semantic_threshold}")
        
        if completion_status['has_progress']:
            print(f"Resuming from previous session...")
        
        # Early exit for fully completed files
        if remaining_pages == 0:
            print("All pages already processed! Output file is up to date.")
            doc.close()  # Clean up PDF resource
            
            # End database session if available
            if self.db_logger and session_id:
                self.db_logger.end_session(session_id, 'completed', completed_pages, 0)
            
            return str(final_output_file)
        
        # Initialize final output file (only if starting fresh)
        if not final_output_file.exists():
            print(f"Initializing output file: {final_output_file}")
            self.content_aggregator.initialize_output_file(final_output_file, book_name)
        else:
            print(f"Resuming with existing output file: {final_output_file}")
        
        # Create file progress context for display
        file_context = ""
        if file_progress:
            file_idx, total_files = file_progress
            file_context = f"File {file_idx}/{total_files} - "
        
        # Load full progress only if needed for processing (lazy loading)
        progress = None
        
        # Process pages with incremental saving
        progress_desc = f"{file_context}Pages"
        with tqdm(total=num_pages, initial=completed_pages, 
                  desc=progress_desc, unit="page", ncols=100) as pbar:
            pages_processed_this_session = 0
            for i in pages_to_process:
                page_num = i + 1  # Convert to 1-based
                current_remaining = remaining_pages - pages_processed_this_session
                
                # Fast check if already processed using the set  
                if page_num in completed_page_set:
                    pbar.update(1)
                    # Keep description stable for completed pages
                    continue
                
                # Lazy load full progress only when we need to process a page
                if progress is None:
                    progress = self.progress_manager.load_page_progress(str(progress_file))
                
                try:
                    # Use stable description - tqdm will show current progress automatically
                    pbar.set_description(f"Processing pages ({current_remaining} remaining)")
                    
                    # Render page as image
                    page = doc[i]
                    pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
                    
                    # Convert to PIL Image
                    mode = "RGBA" if pix.alpha else "RGB"
                    img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
                    
                    # Step 1: Combined pre-assessment (legibility + semantic prediction)
                    assessment_result = self.ocr_engine.combined_pre_assessment(
                        img, page_num, legibility_threshold, semantic_threshold
                    )
                    
                    page_content = ""
                    ocr_confidence = None
                    semantic_score = assessment_result.expected_semantic_quality
                    
                    if assessment_result.should_process:
                        # Step 2: Extract text (only if pre-assessment passed)
                        ocr_result = self.ocr_engine.extract_text(img, page_num)
                        
                        # Use raw OCR text directly (no cleaning)
                        page_content = ocr_result.extracted_text
                        ocr_confidence = ocr_result.confidence
                        status = 'completed'
                        
                        # Immediately append to final file using content aggregator
                        self.content_aggregator.append_page_content(final_output_file, page_num, page_content)
                        
                    else:
                        # Don't create MD file for failed assessments - details are in CSV
                        if assessment_result.legibility_score < legibility_threshold:
                            status = 'illegible'
                        else:
                            status = 'semantically_invalid'
                    
                    # Update progress
                    total_processing_time = assessment_result.processing_time
                    if 'ocr_result' in locals():
                        total_processing_time += ocr_result.processing_time
                    
                    page_progress = PageProgress(
                        page_num=page_num,
                        status=status,
                        legibility_score=assessment_result.legibility_score,
                        semantic_score=semantic_score,
                        ocr_confidence=ocr_confidence,
                        processing_time=total_processing_time,
                        error_message=None,
                        timestamp=datetime.now().isoformat(),
                        text_clarity=assessment_result.text_clarity,
                        image_quality=assessment_result.image_quality,
                        ocr_prediction=assessment_result.ocr_prediction,
                        semantic_prediction=assessment_result.semantic_prediction,
                        visible_text_sample=assessment_result.visible_text_sample,
                        language_detected=assessment_result.language_detected,
                        issues_found=', '.join(assessment_result.issues_found)
                    )
                    progress[page_num] = page_progress
                    
                    # Save progress incrementally (faster than rewriting entire file)
                    self.progress_manager.append_page_progress(page_progress, str(progress_file))
                    
                    # Log to database if available
                    if self.db_logger and session_id:
                        self.db_logger.log_processing_complete(
                            session_id=session_id,
                            file_path=pdf_path,
                            page_number=page_num,
                            status=status,
                            legibility_score=assessment_result.legibility_score,
                            semantic_score=semantic_score,
                            ocr_confidence=ocr_confidence,
                            processing_time=total_processing_time,
                            text_clarity=assessment_result.text_clarity,
                            image_quality=assessment_result.image_quality,
                            ocr_prediction=assessment_result.ocr_prediction,
                            semantic_prediction=assessment_result.semantic_prediction,
                            visible_text_sample=assessment_result.visible_text_sample,
                            language_detected=assessment_result.language_detected,
                            issues_found=', '.join(assessment_result.issues_found)
                        )
                    
                    pbar.update(1)
                    pages_processed_this_session += 1
                    
                    # Update description with current page info (less frequently to avoid flicker)
                    if pages_processed_this_session % 5 == 0 or (remaining_pages - pages_processed_this_session) <= 3:
                        current_remaining = remaining_pages - pages_processed_this_session
                        pbar.set_description(f"{file_context}Pages ({current_remaining} remaining)")
                    
                    # Rate limiting
                    time.sleep(1)
                    
                except Exception as e:
                    error_msg = f"Error processing page {page_num}: {str(e)}"
                    # Use tqdm.write for error messages to avoid interfering with progress bar
                    pbar.write(f"\n{error_msg}")
                    
                    # Immediately append error to final file using content aggregator
                    self.content_aggregator.append_error_content(final_output_file, page_num, error_msg)
                    
                    error_page_progress = PageProgress(
                        page_num=page_num,
                        status='error',
                        legibility_score=None,
                        semantic_score=None,
                        ocr_confidence=None,
                        processing_time=0,
                        error_message=str(e),
                        timestamp=datetime.now().isoformat(),
                        text_clarity=None,
                        image_quality=None,
                        ocr_prediction=None,
                        semantic_prediction=None,
                        visible_text_sample=None,
                        language_detected=None,
                        issues_found=None
                    )
                    progress[page_num] = error_page_progress
                    
                    # Save progress incrementally
                    self.progress_manager.append_page_progress(error_page_progress, str(progress_file))
                    
                    # Log error to database if available
                    if self.db_logger and session_id:
                        self.db_logger.log_processing_complete(
                            session_id=session_id,
                            file_path=pdf_path,
                            page_number=page_num,
                            status='error',
                            error_message=str(e)
                        )
                        # Also log the error specifically
                        import traceback
                        self.db_logger.log_error(
                            error_type="PageProcessingError",
                            error_message=str(e),
                            stack_trace=traceback.format_exc(),
                            file_path=pdf_path,
                            function_name="process_pdf",
                            severity="medium",
                            session_id=session_id
                        )
                    
                    pbar.update(1)
                    pages_processed_this_session += 1
                    
                    # Update description with current page info (less frequently to avoid flicker)
                    if pages_processed_this_session % 5 == 0 or (remaining_pages - pages_processed_this_session) <= 3:
                        current_remaining = remaining_pages - pages_processed_this_session
                        pbar.set_description(f"{file_context}Pages ({current_remaining} remaining)")
        
        doc.close()
        
        # Final file already written incrementally during processing
        
        # Print summary
        completed = sum(1 for p in progress.values() if p.status == 'completed')
        illegible = sum(1 for p in progress.values() if p.status == 'illegible')
        semantic_invalid = sum(1 for p in progress.values() if p.status == 'semantically_invalid')
        errors = sum(1 for p in progress.values() if p.status == 'error')
        total_processed = len(progress)
        
        print(f"\n{'='*60}")
        print(f"PDF PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Total pages processed: {total_processed}/{num_pages}")
        print(f"Success rate: {(completed/total_processed)*100:.1f}%" if total_processed > 0 else "Success rate: 0%")
        print(f"")
        print(f"Results breakdown:")
        print(f"  ✓ Successful OCR: {completed} pages")
        print(f"  👁 Visually illegible: {illegible} pages") 
        print(f"  🧠 Semantically invalid: {semantic_invalid} pages")
        print(f"  ❌ Error pages: {errors} pages")
        print(f"")
        print(f"Files generated:")
        print(f"  📊 Progress log: {progress_file}")
        print(f"  📄 Final output: {final_output_file}")
        
        if pages_processed_this_session > 0:
            print(f"")
            print(f"This session processed: {pages_processed_this_session} new pages")
        
        # End database session if available
        if self.db_logger and session_id:
            self.db_logger.update_session(
                session_id,
                total_files=len(progress),
                completed_files=completed,
                failed_files=illegible + semantic_invalid + errors
            )
            self.db_logger.end_session(session_id, 'completed')
        
        return str(final_output_file)


class ImageProcessor:
    """Handles single image processing."""
    
    def __init__(self, ocr_engine: GeminiOCREngine, db_logger=None, logs_dir: str = './logs'):
        self.ocr_engine = ocr_engine
        self.db_logger = db_logger
        self.logs_dir = logs_dir
        self.progress_manager = ProgressManager(db_logger=db_logger, logs_dir=logs_dir)
        self.image_handler = ImageHandler()
        self.content_aggregator = ContentAggregator()
    
    def process_single_image(self, image_path: str, output_dir: str, 
                           legibility_threshold: float = 0.5, semantic_threshold: float = 0.6) -> str:
        """
        Process a single image file with enhanced OCR and legibility detection.
        
        Args:
            image_path: Path to the image file
            output_dir: Directory to save output
            legibility_threshold: Minimum legibility score to attempt OCR
            
        Returns:
            Path to the processed markdown file
        """
        image_file = Path(image_path)
        if not image_file.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Setup output paths
        image_name = image_file.stem
        output_dir_path = Path(output_dir) / f"{image_name}_image"
        output_dir_path.mkdir(parents=True, exist_ok=True)
        
        progress_file = output_dir_path / f"{image_name}_progress.csv"
        final_output_file = output_dir_path / f"{image_name}_processed.md"
        
        # Check if already processed
        if final_output_file.exists():
            print(f"Image already processed: {final_output_file}")
            return str(final_output_file)
        
        print(f"Processing single image: {image_path}")
        print(f"Legibility threshold: {legibility_threshold}")
        print(f"Semantic threshold: {semantic_threshold}")
        
        try:
            # Load image
            print("Loading image...")
            img = Image.open(image_path)
            
            # Convert to RGB if necessary
            if img.mode not in ('RGB', 'RGBA'):
                img = img.convert('RGB')
            
            # Step 1: Combined pre-assessment
            print("Pre-assessing image quality and content...")
            assessment_result = self.ocr_engine.combined_pre_assessment(
                img, 1, legibility_threshold, semantic_threshold
            )
            
            image_content = ""
            ocr_confidence = None
            semantic_score = assessment_result.expected_semantic_quality
            status = ""
            
            if assessment_result.should_process:
                try:
                    # Step 2: Extract text (only if pre-assessment passed)
                    print("Extracting text...")
                    ocr_result = self.ocr_engine.extract_text(img, 1)
                    
                    # Use raw OCR text directly (no cleaning)
                    image_content = ocr_result.extracted_text
                    ocr_confidence = ocr_result.confidence
                    status = 'completed'
                    error_message = None
                    
                    # Only save MD file for successful OCR
                    with open(final_output_file, 'w', encoding='utf-8') as f:
                        f.write(image_content)
                    
                    print(f"✅ Text extraction successful (OCR confidence: {ocr_confidence:.2f}, Pre-assessed quality: {semantic_score:.2f})")
                    
                except Exception as ocr_error:
                    # OCR extraction failed - don't create MD file
                    print(f"❌ OCR extraction failed: {str(ocr_error)}")
                    ocr_confidence = None
                    status = 'error'
                    error_message = str(ocr_error)
                    # Don't create MD file for OCR failures
            
            else:
                # Don't create MD file for failed assessments - details are in CSV
                ocr_confidence = None
                error_message = None
                if assessment_result.legibility_score < legibility_threshold:
                    status = 'illegible'
                    print(f"❌ Image marked as illegible (score: {assessment_result.legibility_score:.2f})")
                else:
                    status = 'semantically_invalid'
                    print(f"⚠️ Image has poor expected semantic quality (score: {assessment_result.expected_semantic_quality:.2f})")
            
            # Calculate total processing time
            total_processing_time = assessment_result.processing_time
            if status == 'completed' and 'ocr_result' in locals():
                total_processing_time += ocr_result.processing_time
            
            # Save progress record
            progress = {
                image_file.name: ImageProgress(
                    file_path=image_file.name,
                    status=status,
                    legibility_score=assessment_result.legibility_score,
                    semantic_score=semantic_score,
                    ocr_confidence=ocr_confidence,
                    processing_time=total_processing_time,
                    error_message=error_message,
                    timestamp=datetime.now().isoformat(),
                    text_clarity=assessment_result.text_clarity,
                    image_quality=assessment_result.image_quality,
                    ocr_prediction=assessment_result.ocr_prediction,
                    semantic_prediction=assessment_result.semantic_prediction,
                    visible_text_sample=assessment_result.visible_text_sample,
                    language_detected=assessment_result.language_detected,
                    issues_found=', '.join(assessment_result.issues_found)
                )
            }
            self.progress_manager.save_image_progress(progress, str(progress_file))
            
            print(f"\nProcessing completed:")
            print(f"- Status: {status}")
            print(f"- Legibility score: {assessment_result.legibility_score:.2f}")
            if semantic_score:
                print(f"- Expected semantic quality: {semantic_score:.2f}")
            if ocr_confidence:
                print(f"- OCR confidence: {ocr_confidence:.2f}")
            print(f"- Progress saved to: {progress_file}")
            if status == 'completed':
                print(f"- Output file: {final_output_file}")
                return str(final_output_file)
            else:
                print(f"- No output file created (status: {status})")
                return ""
            
        except Exception as e:
            error_msg = f"Error processing {image_path}: {str(e)}"
            print(f"❌ {error_msg}")
            
            # For pre-assessment or other errors, still create error MD file
            error_content = f"Error processing {image_file.name}:\n\n{error_msg}"
            with open(final_output_file, 'w', encoding='utf-8') as f:
                f.write(error_content)
            
            # Save error progress
            progress = {
                image_file.name: ImageProgress(
                    file_path=image_file.name,
                    status='error',
                    legibility_score=None,
                    semantic_score=None,
                    ocr_confidence=None,
                    processing_time=0,
                    error_message=str(e),
                    timestamp=datetime.now().isoformat(),
                    text_clarity=None,
                    image_quality=None,
                    ocr_prediction=None,
                    semantic_prediction=None,
                    visible_text_sample=None,
                    language_detected=None,
                    issues_found=None
                )
            }
            self.progress_manager.save_image_progress(progress, str(progress_file))
            
            return str(final_output_file)


class ImageDirectoryProcessor:
    """Handles batch processing of image directories."""
    
    def __init__(self, ocr_engine: GeminiOCREngine, db_logger=None, logs_dir: str = './logs'):
        self.ocr_engine = ocr_engine
        self.db_logger = db_logger
        self.logs_dir = logs_dir
        self.progress_manager = ProgressManager(db_logger=db_logger, logs_dir=logs_dir)
        self.image_handler = ImageHandler()
        self.content_aggregator = ContentAggregator()
    
    def process_images(self, input_dir: str, output_dir: str, 
                      legibility_threshold: float = 0.5, semantic_threshold: float = 0.6) -> str:
        """
        Process all image files in a directory with enhanced OCR and legibility detection.
        
        Args:
            input_dir: Directory containing image files
            output_dir: Directory to save output
            legibility_threshold: Minimum legibility score to attempt OCR
            
        Returns:
            Path to the final processed markdown file
        """
        # Find all image files
        print(f"Scanning for image files in: {input_dir}")
        image_files = find_image_files(input_dir)
        
        if not image_files:
            print("No supported image files found!")
            return ""
        
        print(f"Found {len(image_files)} image files")
        
        # Setup output paths
        dir_name = Path(input_dir).name
        output_dir_path = Path(output_dir) / f"{dir_name}_images"
        output_dir_path.mkdir(parents=True, exist_ok=True)
        
        progress_file = output_dir_path / f"{dir_name}_images_progress.csv"
        final_output_file = output_dir_path / f"{dir_name}_images_processed.md"
        
        # Use fast resume method to get completion status
        total_files = len(image_files)
        completion_status = self.progress_manager.get_image_completion_status_fast(
            str(progress_file), total_files
        )
        
        completed_count = completion_status['completed_count']
        remaining_count = completion_status['remaining_count']
        completed_file_set = completion_status['completed_files']
        
        print(f"Processing {total_files} images")
        print(f"Progress: {completed_count}/{total_files} completed ({remaining_count} remaining)")
        print(f"Completion: {(completed_count/total_files)*100:.1f}%")
        print(f"Legibility threshold: {legibility_threshold}")
        print(f"Semantic threshold: {semantic_threshold}")
        
        if completion_status['has_progress']:
            print(f"Resuming from previous session...")
        
        # Early exit for fully completed processing
        if remaining_count == 0:
            print("All images already processed! Generating final output file...")
            # We still need to generate the final output file, so don't return early here
            # but we can skip the processing loop entirely
        
        # Process images
        all_content = {}
        
        # Load full progress only if needed for processing (lazy loading)
        progress = None
        
        with tqdm(total=len(image_files), initial=completed_count, 
                  desc=f"Processing images", unit="image") as pbar:
            files_processed_this_session = 0
            for idx, (file_path, relative_path) in enumerate(image_files, 1):
                current_remaining = remaining_count - files_processed_this_session
                
                # Fast check if already processed using the set
                if relative_path in completed_file_set:
                    pbar.update(1)
                    # Note: Content will be loaded later during final file generation
                    continue
                
                # Lazy load full progress only when we need to process a file
                if progress is None:
                    progress = self.progress_manager.load_image_progress(str(progress_file))
                
                try:
                    # Use stable description - tqdm shows progress automatically
                    pbar.set_description(f"Processing images ({current_remaining} remaining)")
                    
                    # Load image
                    img = Image.open(file_path)
                    
                    # Convert to RGB if necessary
                    if img.mode not in ('RGB', 'RGBA'):
                        img = img.convert('RGB')
                    
                    # Step 1: Combined pre-assessment
                    assessment_result = self.ocr_engine.combined_pre_assessment(
                        img, idx, legibility_threshold, semantic_threshold
                    )
                    
                    image_content = ""
                    ocr_confidence = None
                    semantic_score = assessment_result.expected_semantic_quality
                    error_message = None
                    
                    if assessment_result.should_process:
                        try:
                            # Step 2: Extract text (only if pre-assessment passed)
                            ocr_result = self.ocr_engine.extract_text(img, idx)
                            
                            # Use raw OCR text directly (no cleaning)
                            image_content = ocr_result.extracted_text
                            ocr_confidence = ocr_result.confidence
                            status = 'completed'
                            
                            # Only save MD file for successful OCR
                            safe_filename = get_safe_filename(relative_path)
                            image_file = output_dir_path / f"{safe_filename}.md"
                            with open(image_file, 'w', encoding='utf-8') as f:
                                f.write(image_content)
                            
                            all_content[relative_path] = image_content
                            
                        except Exception as ocr_error:
                            # OCR extraction failed - treat as error, don't create MD file
                            print(f"\nOCR extraction failed for {relative_path}: {str(ocr_error)}")
                            ocr_confidence = None
                            status = 'error'
                            error_message = str(ocr_error)
                        
                    else:
                        # Don't create MD file for failed assessments - details are in CSV
                        if assessment_result.legibility_score < legibility_threshold:
                            status = 'illegible'
                        else:
                            status = 'semantically_invalid'
                    
                    # Calculate total processing time
                    total_processing_time = assessment_result.processing_time
                    if status == 'completed' and 'ocr_result' in locals():
                        total_processing_time += ocr_result.processing_time
                    
                    # Update progress
                    image_progress = ImageProgress(
                        file_path=relative_path,
                        status=status,
                        legibility_score=assessment_result.legibility_score,
                        semantic_score=semantic_score,
                        ocr_confidence=ocr_confidence,
                        processing_time=total_processing_time,
                        error_message=error_message,
                        timestamp=datetime.now().isoformat(),
                        text_clarity=assessment_result.text_clarity,
                        image_quality=assessment_result.image_quality,
                        ocr_prediction=assessment_result.ocr_prediction,
                        semantic_prediction=assessment_result.semantic_prediction,
                        visible_text_sample=assessment_result.visible_text_sample,
                        language_detected=assessment_result.language_detected,
                        issues_found=', '.join(assessment_result.issues_found)
                    )
                    progress[relative_path] = image_progress
                    
                    # Save progress incrementally (faster than rewriting entire file)
                    self.progress_manager.append_image_progress(image_progress, str(progress_file))
                    
                    pbar.update(1)
                    files_processed_this_session += 1
                    current_remaining = remaining_count - files_processed_this_session
                    pbar.set_description(f"Processing images ({current_remaining} remaining)")
                    
                    # Rate limiting
                    time.sleep(1)
                    
                except Exception as e:
                    error_msg = f"Error processing {relative_path}: {str(e)}"
                    print(f"\n{error_msg}")
                    
                    # Save error content
                    image_content = f"Error processing {relative_path}:\n\n{error_msg}"
                    safe_filename = get_safe_filename(relative_path)
                    image_file = output_dir_path / f"{safe_filename}.md"
                    with open(image_file, 'w', encoding='utf-8') as f:
                        f.write(image_content)
                    
                    all_content[relative_path] = image_content
                    
                    error_progress = ImageProgress(
                        file_path=relative_path,
                        status='error',
                        legibility_score=None,
                        semantic_score=None,
                        ocr_confidence=None,
                        processing_time=0,
                        error_message=str(e),
                        timestamp=datetime.now().isoformat(),
                        text_clarity=None,
                        image_quality=None,
                        ocr_prediction=None,
                        semantic_prediction=None,
                        visible_text_sample=None,
                        language_detected=None,
                        issues_found=None
                    )
                    progress[relative_path] = error_progress
                    
                    # Save progress incrementally
                    self.progress_manager.append_image_progress(error_progress, str(progress_file))
                    pbar.update(1)
                    files_processed_this_session += 1
                    current_remaining = remaining_count - files_processed_this_session
                    pbar.set_description(f"Processing images ({current_remaining} remaining)")
        
        # Combine all images into final markdown file
        print("\nCombining images into final markdown file...")
        
        # Helper function to load content on-demand
        def load_content_for_file(relative_path):
            """Load content from file on-demand."""
            if relative_path in all_content:
                return all_content[relative_path]
            
            # Load from file if exists
            safe_filename = get_safe_filename(relative_path)
            image_file = output_dir_path / f"{safe_filename}.md"
            if image_file.exists():
                with open(image_file, 'r', encoding='utf-8') as f:
                    return f.read()
            return ""
        
        # Get all processed files (both in-memory and from completed progress)
        all_processed_files = set(all_content.keys())
        for relative_path, prog in progress.items():
            if prog.status == 'completed':
                all_processed_files.add(relative_path)
        
        # Group by subdirectory for better organization
        grouped_content = {}
        for relative_path in sorted(all_processed_files):
            subdir = str(Path(relative_path).parent) if Path(relative_path).parent != Path('.') else 'root'
            if subdir not in grouped_content:
                grouped_content[subdir] = []
            grouped_content[subdir].append(relative_path)
        
        # Stream content directly to final file
        with open(final_output_file, 'w', encoding='utf-8') as f:
            f.write(f"# {dir_name} - Image OCR Results\n\n")
            
            for subdir in sorted(grouped_content.keys()):
                if subdir != 'root':
                    f.write(f"## {subdir}\n\n")
                
                for relative_path in grouped_content[subdir]:
                    content = load_content_for_file(relative_path)
                    if content:  # Only write non-empty content
                        f.write(content)
                        f.write("\n\n---\n\n")  # Content separator
        
        # Print summary
        completed = sum(1 for p in progress.values() if p.status == 'completed')
        illegible = sum(1 for p in progress.values() if p.status == 'illegible')
        semantic_invalid = sum(1 for p in progress.values() if p.status == 'semantically_invalid')
        errors = sum(1 for p in progress.values() if p.status == 'error')
        
        total_processed = len(progress)
        
        print(f"\n{'='*60}")
        print(f"PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Total files processed: {total_processed}/{total_files}")
        print(f"Success rate: {(completed/total_processed)*100:.1f}%" if total_processed > 0 else "Success rate: 0%")
        print(f"")
        print(f"Results breakdown:")
        print(f"  ✓ Successful OCR: {completed} images")
        print(f"  👁 Visually illegible: {illegible} images") 
        print(f"  🧠 Semantically invalid: {semantic_invalid} images")
        print(f"  ❌ Error images: {errors} images")
        print(f"")
        print(f"Files generated:")
        print(f"  📊 Progress log: {progress_file}")
        print(f"  📄 Final output: {final_output_file}")
        
        if files_processed_this_session > 0:
            print(f"")
            print(f"This session processed: {files_processed_this_session} new files")
        
        return str(final_output_file)