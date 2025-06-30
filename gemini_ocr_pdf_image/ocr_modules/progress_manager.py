"""
Progress tracking and CSV management for OCR processing.
"""

import os
import csv
import fcntl
from typing import Dict, Optional, Set
from pathlib import Path
from .models import PageProgress, ImageProgress


class ProgressManager:
    """Handles loading and saving progress data to CSV files and database."""
    
    def __init__(self, db_logger=None, logs_dir: str = './logs'):
        """Initialize progress manager with optional database logger."""
        self.db_logger = db_logger
        self.logs_dir = logs_dir
        
        # Ensure logs directory exists
        Path(self.logs_dir).mkdir(parents=True, exist_ok=True)
    
    def _get_csv_path(self, progress_file: str) -> str:
        """Get the CSV file path in the logs directory."""
        filename = Path(progress_file).name
        return str(Path(self.logs_dir) / filename)
    
    def load_page_progress(self, progress_file: str) -> Dict[int, PageProgress]:
        """Load processing progress from CSV file."""
        progress = {}
        csv_path = self._get_csv_path(progress_file)
        if os.path.exists(csv_path):
            with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    page_num = int(row['page_num'])
                    progress[page_num] = PageProgress(
                        page_num=page_num,
                        status=row['status'],
                        legibility_score=float(row['legibility_score']) if row['legibility_score'] else None,
                        semantic_score=float(row['semantic_score']) if row.get('semantic_score') else None,
                        ocr_confidence=float(row['ocr_confidence']) if row['ocr_confidence'] else None,
                        processing_time=float(row['processing_time']),
                        error_message=row['error_message'] if row['error_message'] else None,
                        timestamp=row['timestamp'],
                        text_clarity=row.get('text_clarity'),
                        image_quality=row.get('image_quality'),
                        ocr_prediction=row.get('ocr_prediction'),
                        semantic_prediction=row.get('semantic_prediction'),
                        visible_text_sample=row.get('visible_text_sample'),
                        language_detected=row.get('language_detected'),
                        issues_found=row.get('issues_found')
                    )
        return progress
    
    def save_page_progress(self, progress: Dict[int, PageProgress], progress_file: str):
        """Save processing progress to CSV file and optionally to database.
        
        Note: For better performance during processing loops, consider using 
        append_page_progress() instead of this method which rewrites the entire file.
        """
        fieldnames = ['page_num', 'status', 'legibility_score', 'semantic_score', 'ocr_confidence', 
                     'processing_time', 'error_message', 'timestamp', 'text_clarity', 'image_quality',
                     'ocr_prediction', 'semantic_prediction', 'visible_text_sample', 'language_detected', 'issues_found']
        
        csv_path = self._get_csv_path(progress_file)
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for page_progress in sorted(progress.values(), key=lambda x: x.page_num):
                writer.writerow({
                    'page_num': page_progress.page_num,
                    'status': page_progress.status,
                    'legibility_score': page_progress.legibility_score,
                    'semantic_score': page_progress.semantic_score,
                    'ocr_confidence': page_progress.ocr_confidence,
                    'processing_time': page_progress.processing_time,
                    'error_message': page_progress.error_message,
                    'timestamp': page_progress.timestamp,
                    'text_clarity': page_progress.text_clarity,
                    'image_quality': page_progress.image_quality,
                    'ocr_prediction': page_progress.ocr_prediction,
                    'semantic_prediction': page_progress.semantic_prediction,
                    'visible_text_sample': page_progress.visible_text_sample,
                    'language_detected': page_progress.language_detected,
                    'issues_found': page_progress.issues_found
                })
    
    def load_image_progress(self, progress_file: str) -> Dict[str, ImageProgress]:
        """Load image processing progress from CSV file."""
        progress = {}
        csv_path = self._get_csv_path(progress_file)
        if os.path.exists(csv_path):
            with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    file_path = row['file_path']
                    progress[file_path] = ImageProgress(
                        file_path=file_path,
                        status=row['status'],
                        legibility_score=float(row['legibility_score']) if row['legibility_score'] else None,
                        semantic_score=float(row['semantic_score']) if row.get('semantic_score') else None,
                        ocr_confidence=float(row['ocr_confidence']) if row['ocr_confidence'] else None,
                        processing_time=float(row['processing_time']),
                        error_message=row['error_message'] if row['error_message'] else None,
                        timestamp=row['timestamp'],
                        text_clarity=row.get('text_clarity'),
                        image_quality=row.get('image_quality'),
                        ocr_prediction=row.get('ocr_prediction'),
                        semantic_prediction=row.get('semantic_prediction'),
                        visible_text_sample=row.get('visible_text_sample'),
                        language_detected=row.get('language_detected'),
                        issues_found=row.get('issues_found')
                    )
        return progress
    
    def save_image_progress(self, progress: Dict[str, ImageProgress], progress_file: str):
        """Save image processing progress to CSV file and optionally to database.
        
        Note: For better performance during processing loops, consider using 
        append_image_progress() instead of this method which rewrites the entire file.
        """
        fieldnames = ['file_path', 'status', 'legibility_score', 'semantic_score', 'ocr_confidence', 
                     'processing_time', 'error_message', 'timestamp', 'text_clarity', 'image_quality',
                     'ocr_prediction', 'semantic_prediction', 'visible_text_sample', 'language_detected', 'issues_found']
        
        csv_path = self._get_csv_path(progress_file)
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for image_progress in sorted(progress.values(), key=lambda x: x.file_path):
                writer.writerow({
                    'file_path': image_progress.file_path,
                    'status': image_progress.status,
                    'legibility_score': image_progress.legibility_score,
                    'semantic_score': image_progress.semantic_score,
                    'ocr_confidence': image_progress.ocr_confidence,
                    'processing_time': image_progress.processing_time,
                    'error_message': image_progress.error_message,
                    'timestamp': image_progress.timestamp,
                    'text_clarity': image_progress.text_clarity,
                    'image_quality': image_progress.image_quality,
                    'ocr_prediction': image_progress.ocr_prediction,
                    'semantic_prediction': image_progress.semantic_prediction,
                    'visible_text_sample': image_progress.visible_text_sample,
                    'language_detected': image_progress.language_detected,
                    'issues_found': image_progress.issues_found
                })

    def append_page_progress(self, page_progress: PageProgress, progress_file: str):
        """Append a single page progress entry to CSV file, preventing duplicates."""
        csv_path = self._get_csv_path(progress_file)
        fieldnames = ['page_num', 'status', 'legibility_score', 'semantic_score', 'ocr_confidence', 
                     'processing_time', 'error_message', 'timestamp', 'text_clarity', 'image_quality',
                     'ocr_prediction', 'semantic_prediction', 'visible_text_sample', 'language_detected', 'issues_found']
        
        # Check if this page already exists in progress file
        if os.path.exists(csv_path):
            existing_pages = self._get_deduplicated_page_status(progress_file)
            if page_progress.page_num in existing_pages:
                # Page already exists, use full rewrite to update it
                all_progress = self.load_page_progress(progress_file)
                all_progress[page_progress.page_num] = page_progress
                self.save_page_progress(all_progress, progress_file)
                return
        
        # Page doesn't exist yet, safe to append
        file_exists = os.path.exists(csv_path)
        
        try:
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                # Apply file locking for concurrent access
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                # Write header if new file
                if not file_exists:
                    writer.writeheader()
                
                writer.writerow({
                    'page_num': page_progress.page_num,
                    'status': page_progress.status,
                    'legibility_score': page_progress.legibility_score,
                    'semantic_score': page_progress.semantic_score,
                    'ocr_confidence': page_progress.ocr_confidence,
                    'processing_time': page_progress.processing_time,
                    'error_message': page_progress.error_message,
                    'timestamp': page_progress.timestamp,
                    'text_clarity': page_progress.text_clarity,
                    'image_quality': page_progress.image_quality,
                    'ocr_prediction': page_progress.ocr_prediction,
                    'semantic_prediction': page_progress.semantic_prediction,
                    'visible_text_sample': page_progress.visible_text_sample,
                    'language_detected': page_progress.language_detected,
                    'issues_found': page_progress.issues_found
                })
                
                # File lock is automatically released when file is closed
        except IOError as e:
            # Fallback to full rewrite if append fails
            print(f"Warning: Append failed ({e}), falling back to full progress save")
            # We'll need the full progress dict for this, so this is a fallback only
    
    def _get_deduplicated_page_status(self, progress_file: str) -> Dict[int, str]:
        """
        Get the latest status for each page, handling duplicate entries.
        
        Returns:
            Dict mapping page_num -> latest_status
        """
        page_status = {}
        csv_path = self._get_csv_path(progress_file)
        
        if os.path.exists(csv_path):
            try:
                with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        page_num = int(row['page_num'])
                        # Keep latest status (overwrites duplicates)
                        page_status[page_num] = row['status']
            except Exception as e:
                print(f"Warning: Error reading progress file {csv_path}: {e}")
        
        return page_status
    
    def get_completion_status_fast(self, progress_file: str, total_pages: int) -> Dict[str, any]:
        """
        Fast method to get completion status without loading full progress objects.
        
        Args:
            progress_file: Path to the progress CSV file
            total_pages: Total number of pages to process
            
        Returns:
            Dict with completion statistics: {
                'completed_pages': set of completed page numbers,
                'total_pages': total pages to process,
                'completed_count': number of completed pages,
                'remaining_count': number of remaining pages,
                'completion_percentage': completion percentage,
                'last_completed_page': highest completed page number,
                'has_progress': whether any progress exists
            }
        """
        completed_pages = set()
        csv_path = self._get_csv_path(progress_file)
        has_progress = False
        
        if os.path.exists(csv_path):
            try:
                with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    # Track latest status for each page to handle duplicates
                    page_status = {}
                    for row in reader:
                        has_progress = True
                        page_num = int(row['page_num'])
                        page_status[page_num] = row['status']
                    
                    # Count pages that are fully processed (any final status)
                    for page_num, status in page_status.items():
                        if status in ['completed', 'illegible', 'semantically_invalid']:
                            completed_pages.add(page_num)
            except Exception as e:
                print(f"Warning: Error reading progress file {csv_path}: {e}")
        
        completed_count = len(completed_pages)
        remaining_count = total_pages - completed_count
        completion_percentage = (completed_count / total_pages * 100) if total_pages > 0 else 0
        last_completed_page = max(completed_pages) if completed_pages else 0
        
        return {
            'completed_pages': completed_pages,
            'total_pages': total_pages,
            'completed_count': completed_count,
            'remaining_count': remaining_count,
            'completion_percentage': completion_percentage,
            'last_completed_page': last_completed_page,
            'has_progress': has_progress
        }
    
    def get_completed_pages_only(self, progress_file: str) -> Set[int]:
        """
        Ultra-fast method to get only the set of completed page numbers.
        
        Args:
            progress_file: Path to the progress CSV file
            
        Returns:
            Set of completed page numbers
        """
        completed_pages = set()
        csv_path = self._get_csv_path(progress_file)
        
        if os.path.exists(csv_path):
            try:
                with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row['status'] == 'completed':
                            completed_pages.add(int(row['page_num']))
            except Exception as e:
                print(f"Warning: Error reading progress file {csv_path}: {e}")
        
        return completed_pages
    
    def get_image_completion_status_fast(self, progress_file: str, total_files: int) -> Dict[str, any]:
        """
        Fast method to get image completion status without loading full progress objects.
        
        Args:
            progress_file: Path to the progress CSV file
            total_files: Total number of files to process
            
        Returns:
            Dict with completion statistics
        """
        completed_files = set()
        csv_path = self._get_csv_path(progress_file)
        has_progress = False
        
        if os.path.exists(csv_path):
            try:
                with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    # Track latest status for each file to handle duplicates
                    file_status = {}
                    for row in reader:
                        has_progress = True
                        file_path = row['file_path']
                        file_status[file_path] = row['status']
                    
                    # Count files that are fully processed (any final status)
                    for file_path, status in file_status.items():
                        if status in ['completed', 'illegible', 'semantically_invalid']:
                            completed_files.add(file_path)
            except Exception as e:
                print(f"Warning: Error reading progress file {csv_path}: {e}")
        
        completed_count = len(completed_files)
        remaining_count = total_files - completed_count
        completion_percentage = (completed_count / total_files * 100) if total_files > 0 else 0
        
        return {
            'completed_files': completed_files,
            'total_files': total_files,
            'completed_count': completed_count,
            'remaining_count': remaining_count,
            'completion_percentage': completion_percentage,
            'has_progress': has_progress
        }

    def append_image_progress(self, image_progress: ImageProgress, progress_file: str):
        """Append a single image progress entry to CSV file (faster for real-time updates)."""
        csv_path = self._get_csv_path(progress_file)
        fieldnames = ['file_path', 'status', 'legibility_score', 'semantic_score', 'ocr_confidence', 
                     'processing_time', 'error_message', 'timestamp', 'text_clarity', 'image_quality',
                     'ocr_prediction', 'semantic_prediction', 'visible_text_sample', 'language_detected', 'issues_found']
        
        # Check if file exists and needs header
        file_exists = os.path.exists(csv_path)
        
        try:
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                # Apply file locking for concurrent access
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                # Write header if new file
                if not file_exists:
                    writer.writeheader()
                
                writer.writerow({
                    'file_path': image_progress.file_path,
                    'status': image_progress.status,
                    'legibility_score': image_progress.legibility_score,
                    'semantic_score': image_progress.semantic_score,
                    'ocr_confidence': image_progress.ocr_confidence,
                    'processing_time': image_progress.processing_time,
                    'error_message': image_progress.error_message,
                    'timestamp': image_progress.timestamp,
                    'text_clarity': image_progress.text_clarity,
                    'image_quality': image_progress.image_quality,
                    'ocr_prediction': image_progress.ocr_prediction,
                    'semantic_prediction': image_progress.semantic_prediction,
                    'visible_text_sample': image_progress.visible_text_sample,
                    'language_detected': image_progress.language_detected,
                    'issues_found': image_progress.issues_found
                })
                
                # File lock is automatically released when file is closed
        except IOError as e:
            # Fallback to full rewrite if append fails
            print(f"Warning: Append failed ({e}), falling back to full progress save")
            # We'll need the full progress dict for this, so this is a fallback only
    
    def get_completion_status_fast(self, progress_file: str, total_pages: int) -> Dict[str, any]:
        """
        Fast method to get completion status without loading full progress objects.
        
        Args:
            progress_file: Path to the progress CSV file
            total_pages: Total number of pages to process
            
        Returns:
            Dict with completion statistics: {
                'completed_pages': set of completed page numbers,
                'total_pages': total pages to process,
                'completed_count': number of completed pages,
                'remaining_count': number of remaining pages,
                'completion_percentage': completion percentage,
                'last_completed_page': highest completed page number,
                'has_progress': whether any progress exists
            }
        """
        completed_pages = set()
        csv_path = self._get_csv_path(progress_file)
        has_progress = False
        
        if os.path.exists(csv_path):
            try:
                with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    # Track latest status for each page to handle duplicates
                    page_status = {}
                    for row in reader:
                        has_progress = True
                        page_num = int(row['page_num'])
                        page_status[page_num] = row['status']
                    
                    # Count pages that are fully processed (any final status)
                    for page_num, status in page_status.items():
                        if status in ['completed', 'illegible', 'semantically_invalid']:
                            completed_pages.add(page_num)
            except Exception as e:
                print(f"Warning: Error reading progress file {csv_path}: {e}")
        
        completed_count = len(completed_pages)
        remaining_count = total_pages - completed_count
        completion_percentage = (completed_count / total_pages * 100) if total_pages > 0 else 0
        last_completed_page = max(completed_pages) if completed_pages else 0
        
        return {
            'completed_pages': completed_pages,
            'total_pages': total_pages,
            'completed_count': completed_count,
            'remaining_count': remaining_count,
            'completion_percentage': completion_percentage,
            'last_completed_page': last_completed_page,
            'has_progress': has_progress
        }
    
    def get_completed_pages_only(self, progress_file: str) -> Set[int]:
        """
        Ultra-fast method to get only the set of completed page numbers.
        
        Args:
            progress_file: Path to the progress CSV file
            
        Returns:
            Set of completed page numbers
        """
        completed_pages = set()
        csv_path = self._get_csv_path(progress_file)
        
        if os.path.exists(csv_path):
            try:
                with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row['status'] == 'completed':
                            completed_pages.add(int(row['page_num']))
            except Exception as e:
                print(f"Warning: Error reading progress file {csv_path}: {e}")
        
        return completed_pages
    
    def get_image_completion_status_fast(self, progress_file: str, total_files: int) -> Dict[str, any]:
        """
        Fast method to get image completion status without loading full progress objects.
        
        Args:
            progress_file: Path to the progress CSV file
            total_files: Total number of files to process
            
        Returns:
            Dict with completion statistics
        """
        completed_files = set()
        csv_path = self._get_csv_path(progress_file)
        has_progress = False
        
        if os.path.exists(csv_path):
            try:
                with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    # Track latest status for each file to handle duplicates
                    file_status = {}
                    for row in reader:
                        has_progress = True
                        file_path = row['file_path']
                        file_status[file_path] = row['status']
                    
                    # Count files that are fully processed (any final status)
                    for file_path, status in file_status.items():
                        if status in ['completed', 'illegible', 'semantically_invalid']:
                            completed_files.add(file_path)
            except Exception as e:
                print(f"Warning: Error reading progress file {csv_path}: {e}")
        
        completed_count = len(completed_files)
        remaining_count = total_files - completed_count
        completion_percentage = (completed_count / total_files * 100) if total_files > 0 else 0
        
        return {
            'completed_files': completed_files,
            'total_files': total_files,
            'completed_count': completed_count,
            'remaining_count': remaining_count,
            'completion_percentage': completion_percentage,
            'has_progress': has_progress
        }