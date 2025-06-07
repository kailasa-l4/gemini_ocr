"""
Progress tracking and CSV management for OCR processing.
"""

import os
import csv
from typing import Dict
from .models import PageProgress, ImageProgress


class ProgressManager:
    """Handles loading and saving progress data to CSV files."""
    
    @staticmethod
    def load_page_progress(progress_file: str) -> Dict[int, PageProgress]:
        """Load processing progress from CSV file."""
        progress = {}
        if os.path.exists(progress_file):
            with open(progress_file, 'r', newline='', encoding='utf-8') as f:
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
    
    @staticmethod
    def save_page_progress(progress: Dict[int, PageProgress], progress_file: str):
        """Save processing progress to CSV file."""
        fieldnames = ['page_num', 'status', 'legibility_score', 'semantic_score', 'ocr_confidence', 
                     'processing_time', 'error_message', 'timestamp', 'text_clarity', 'image_quality',
                     'ocr_prediction', 'semantic_prediction', 'visible_text_sample', 'language_detected', 'issues_found']
        
        with open(progress_file, 'w', newline='', encoding='utf-8') as f:
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
    
    @staticmethod
    def load_image_progress(progress_file: str) -> Dict[str, ImageProgress]:
        """Load image processing progress from CSV file."""
        progress = {}
        if os.path.exists(progress_file):
            with open(progress_file, 'r', newline='', encoding='utf-8') as f:
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
    
    @staticmethod
    def save_image_progress(progress: Dict[str, ImageProgress], progress_file: str):
        """Save image processing progress to CSV file."""
        fieldnames = ['file_path', 'status', 'legibility_score', 'semantic_score', 'ocr_confidence', 
                     'processing_time', 'error_message', 'timestamp', 'text_clarity', 'image_quality',
                     'ocr_prediction', 'semantic_prediction', 'visible_text_sample', 'language_detected', 'issues_found']
        
        with open(progress_file, 'w', newline='', encoding='utf-8') as f:
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