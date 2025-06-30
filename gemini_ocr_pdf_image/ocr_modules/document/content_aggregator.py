"""
Content aggregation service for combining OCR results.

This module handles the aggregation of OCR results from multiple sources
into final output files with proper formatting and organization.
"""

from pathlib import Path
from typing import Dict, List, Optional, Set
import logging


class ContentAggregator:
    """
    Service for aggregating and organizing OCR content into final output files.
    
    Handles memory-efficient content streaming, subdirectory organization,
    and proper markdown formatting for final output generation.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the content aggregator.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger('ContentAggregator')
    
    def initialize_output_file(self, output_file: Path, title: str) -> None:
        """
        Initialize an output file with a title header.
        
        Args:
            output_file: Path to the output file
            title: Title for the document
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# {title}\n\n")
    
    def append_page_content(self, output_file: Path, page_num: int, content: str) -> None:
        """
        Append page content to an output file with proper formatting.
        
        Args:
            output_file: Path to the output file
            page_num: Page number for the content
            content: Content to append
        """
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(f"## Page {page_num}\n\n")
            f.write(content)
            f.write("\n\n---\n\n")
    
    def append_error_content(self, output_file: Path, page_num: int, error_message: str) -> None:
        """
        Append error content to an output file.
        
        Args:
            output_file: Path to the output file
            page_num: Page number where the error occurred
            error_message: Error message to append
        """
        error_content = f"Error processing page {page_num}:\n\n{error_message}"
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(f"## Page {page_num} (Error)\n\n")
            f.write(error_content)
            f.write("\n\n---\n\n")
    
    def load_content_from_file(self, file_path: Path) -> str:
        """
        Load content from a file on-demand.
        
        Args:
            file_path: Path to the file to load
            
        Returns:
            File content as string, or empty string if file doesn't exist
        """
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                self.logger.error(f"Failed to load content from {file_path}: {str(e)}")
                return ""
        return ""
    
    def create_image_output_file(self, output_dir: Path, relative_path: str, content: str) -> Path:
        """
        Create an individual output file for an image.
        
        Args:
            output_dir: Output directory path
            relative_path: Relative path of the source image
            content: Content to write
            
        Returns:
            Path to the created output file
        """
        from ..utils import get_safe_filename
        
        safe_filename = get_safe_filename(relative_path)
        output_file = output_dir / f"{safe_filename}.md"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return output_file
    
    def create_error_output_file(self, output_dir: Path, relative_path: str, error_message: str) -> Path:
        """
        Create an error output file for a failed image.
        
        Args:
            output_dir: Output directory path
            relative_path: Relative path of the source image
            error_message: Error message to write
            
        Returns:
            Path to the created error file
        """
        error_content = f"Error processing {relative_path}:\n\n{error_message}"
        return self.create_image_output_file(output_dir, relative_path, error_content)
    
    def get_processed_files_from_progress(self, progress: dict) -> Set[str]:
        """
        Get all processed files from progress dictionary.
        
        Args:
            progress: Progress dictionary with file processing status
            
        Returns:
            Set of relative paths for all processed files
        """
        return {relative_path for relative_path, prog in progress.items() 
                if prog.status == 'completed'}
    
    def group_files_by_subdirectory(self, file_paths: List[str]) -> Dict[str, List[str]]:
        """
        Group file paths by their subdirectory.
        
        Args:
            file_paths: List of relative file paths
            
        Returns:
            Dictionary mapping subdirectory names to lists of file paths
        """
        grouped = {}
        for relative_path in sorted(file_paths):
            subdir = str(Path(relative_path).parent) if Path(relative_path).parent != Path('.') else 'root'
            if subdir not in grouped:
                grouped[subdir] = []
            grouped[subdir].append(relative_path)
        
        return grouped
    
    def create_combined_output_file(self, 
                                  output_file: Path, 
                                  title: str,
                                  all_content: Dict[str, str],
                                  processed_files: Set[str],
                                  output_dir: Path) -> None:
        """
        Create a combined output file from all processed content.
        
        Args:
            output_file: Path to the combined output file
            title: Title for the document
            all_content: Dictionary mapping file paths to their content
            processed_files: Set of all processed file paths
            output_dir: Directory containing individual output files
        """
        def load_content_for_file(relative_path: str) -> str:
            """Load content from file on-demand."""
            if relative_path in all_content:
                return all_content[relative_path]
            
            # Load from individual file if exists
            from ..utils import get_safe_filename
            safe_filename = get_safe_filename(relative_path)
            image_file = output_dir / f"{safe_filename}.md"
            return self.load_content_from_file(image_file)
        
        # Group files by subdirectory
        all_processed_files = set(all_content.keys()) | processed_files
        grouped_content = self.group_files_by_subdirectory(list(all_processed_files))
        
        # Stream content directly to final file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# {title}\n\n")
            
            for subdir in sorted(grouped_content.keys()):
                if subdir != 'root':
                    f.write(f"## {subdir}\n\n")
                
                for relative_path in grouped_content[subdir]:
                    content = load_content_for_file(relative_path)
                    if content:  # Only write non-empty content
                        f.write(content)
                        f.write("\n\n---\n\n")  # Content separator
        
        self.logger.info(f"Created combined output file: {output_file}")
    
    def create_processing_summary(self, progress: dict) -> Dict[str, int]:
        """
        Create a summary of processing results.
        
        Args:
            progress: Progress dictionary with processing status
            
        Returns:
            Dictionary with processing statistics
        """
        completed = sum(1 for p in progress.values() if p.status == 'completed')
        illegible = sum(1 for p in progress.values() if p.status == 'illegible')
        semantic_invalid = sum(1 for p in progress.values() if p.status == 'semantically_invalid')
        errors = sum(1 for p in progress.values() if p.status == 'error')
        total_processed = len(progress)
        
        return {
            'completed': completed,
            'illegible': illegible,
            'semantic_invalid': semantic_invalid,
            'errors': errors,
            'total_processed': total_processed,
            'success_rate': (completed / total_processed) * 100 if total_processed > 0 else 0
        }
    
    def print_processing_summary(self, 
                               summary: Dict[str, int], 
                               total_files: int,
                               progress_file: Path, 
                               output_file: Path,
                               files_processed_this_session: int = 0) -> None:
        """
        Print a formatted processing summary.
        
        Args:
            summary: Processing summary dictionary
            total_files: Total number of files that were supposed to be processed
            progress_file: Path to the progress file
            output_file: Path to the output file
            files_processed_this_session: Number of files processed in current session
        """
        print(f"\n{'='*60}")
        print(f"PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Total files processed: {summary['total_processed']}/{total_files}")
        print(f"Success rate: {summary['success_rate']:.1f}%")
        print(f"")
        print(f"Results breakdown:")
        print(f"  ✓ Successful OCR: {summary['completed']} files")
        print(f"  👁 Visually illegible: {summary['illegible']} files")
        print(f"  🧠 Semantically invalid: {summary['semantic_invalid']} files")
        print(f"  ❌ Error files: {summary['errors']} files")
        print(f"")
        print(f"Files generated:")
        print(f"  📊 Progress log: {progress_file}")
        print(f"  📄 Final output: {output_file}")
        
        if files_processed_this_session > 0:
            print(f"")
            print(f"This session processed: {files_processed_this_session} new files")