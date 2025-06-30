"""
Image document handling service for OCR processing.

This module handles image-specific operations including file loading,
format conversion, and directory scanning operations.
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional, Generator
from PIL import Image

from ..utils import find_image_files, get_safe_filename


class ImageHandler:
    """
    Specialized handler for image file processing operations.
    
    Handles image loading, format conversion, directory scanning,
    and provides a clean interface for image-specific OCR operations.
    """
    
    def __init__(self):
        """Initialize the image handler."""
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    def validate_image_file(self, image_path: str) -> bool:
        """
        Validate if a file is a supported image format.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            True if the file is a supported image format
        """
        return Path(image_path).suffix.lower() in self.supported_formats
    
    def load_image(self, image_path: str) -> Image.Image:
        """
        Load an image file and ensure it's in a compatible format.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            PIL Image object in RGB or RGBA format
            
        Raises:
            Exception: If image cannot be loaded
        """
        try:
            image = Image.open(image_path)
            
            # Convert to RGB if necessary (but preserve RGBA for transparency)
            if image.mode not in ('RGB', 'RGBA'):
                image = image.convert('RGB')
            
            return image
            
        except Exception as e:
            raise Exception(f"Failed to load image {image_path}: {str(e)}")
    
    def scan_directory(self, directory_path: str) -> List[Tuple[str, str]]:
        """
        Scan a directory for image files recursively.
        
        Args:
            directory_path: Path to the directory to scan
            
        Returns:
            List of tuples (absolute_path, relative_path) for each image found
        """
        return find_image_files(directory_path)
    
    def get_image_iterator(self, image_files: List[Tuple[str, str]]) -> Generator[Tuple[str, str, int], None, None]:
        """
        Get an iterator for processing image files.
        
        Args:
            image_files: List of (absolute_path, relative_path) tuples
            
        Yields:
            Tuple of (absolute_path, relative_path, index)
        """
        for idx, (absolute_path, relative_path) in enumerate(image_files, 1):
            yield absolute_path, relative_path, idx
    
    def calculate_progress_summary(self, image_files: List[Tuple[str, str]], progress: dict) -> dict:
        """
        Calculate progress summary for image processing.
        
        Args:
            image_files: List of (absolute_path, relative_path) tuples
            progress: Progress dictionary from progress manager
            
        Returns:
            Dictionary with progress statistics
        """
        total_files = len(image_files)
        completed_count = sum(1 for _, relative_path in image_files 
                            if relative_path in progress and progress[relative_path].status == 'completed')
        remaining_count = total_files - completed_count
        completion_percentage = (completed_count / total_files) * 100 if total_files > 0 else 0
        
        return {
            'total_files': total_files,
            'completed_files': completed_count,
            'remaining_files': remaining_count,
            'completion_percentage': completion_percentage,
            'files_to_process': [relative_path for _, relative_path in image_files]
        }
    
    def create_safe_filename(self, relative_path: str) -> str:
        """
        Create a safe filename for output files.
        
        Args:
            relative_path: Relative path of the image file
            
        Returns:
            Safe filename for use in output files
        """
        return get_safe_filename(relative_path)
    
    def group_by_subdirectory(self, image_files: List[Tuple[str, str]]) -> dict:
        """
        Group image files by their subdirectory for organized output.
        
        Args:
            image_files: List of (absolute_path, relative_path) tuples
            
        Returns:
            Dictionary mapping subdirectory names to lists of relative paths
        """
        grouped = {}
        for _, relative_path in image_files:
            subdir = str(Path(relative_path).parent) if Path(relative_path).parent != Path('.') else 'root'
            if subdir not in grouped:
                grouped[subdir] = []
            grouped[subdir].append(relative_path)
        
        return grouped
    
    def validate_directory(self, directory_path: str) -> bool:
        """
        Validate if a directory exists and is accessible.
        
        Args:
            directory_path: Path to the directory
            
        Returns:
            True if directory is valid and accessible
        """
        path = Path(directory_path)
        return path.exists() and path.is_dir()
    
    def get_directory_stats(self, directory_path: str) -> dict:
        """
        Get statistics about a directory.
        
        Args:
            directory_path: Path to the directory
            
        Returns:
            Dictionary with directory statistics
        """
        if not self.validate_directory(directory_path):
            return {'error': 'Directory not found or not accessible'}
        
        image_files = self.scan_directory(directory_path)
        grouped = self.group_by_subdirectory(image_files)
        
        return {
            'total_images': len(image_files),
            'subdirectories': len(grouped),
            'directory_structure': {subdir: len(files) for subdir, files in grouped.items()}
        }