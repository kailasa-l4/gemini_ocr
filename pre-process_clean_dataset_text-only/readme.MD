# Book Processing Pipeline

A scalable pipeline for processing hundreds of books with multiple files per book, specifically designed for cleaning and preparing spiritual texts for LLM training. This comprehensive suite of tools handles the entire processing workflow from text merging to data analysis.

## Components

This project consists of several integrated tools:

1. **Book Processor** (`book_processor.py`): The core processing engine that merges book files, cleans text, extracts metadata, and creates datasets
2. **Batch Manager** (`batch-manager.py`): A monitoring and management tool for overseeing the processing pipeline
3. **Chunk Viewer** (`chunk-viewer-tui.py`): A text-based UI for visualizing processing in real-time
4. **Output Analyzer** (`output_analyzer.py`): A tool for analyzing and visualizing the quality of processed text
5. **Simple Dashboard** (`simple_dashboard.py`): A lightweight terminal-based dashboard for processing status

## Features

- **Scalable Processing**: Handle hundreds of books with thousands of files
- **Database Tracking**: SQLite-based progress tracking and monitoring
- **Parallel Processing**: Process multiple books and chunks concurrently
- **Intelligent Merging**: Combine multi-file books with correct ordering
- **Clean Text Processing**: Remove artifacts, normalize formatting, and standardize content
- **Metadata Extraction**: Extract structured information about book content
- **Live Monitoring**: Multiple visualization tools for process monitoring
- **Quality Analysis**: Validate and visualize processing effectiveness
- **Dataset Creation**: Generate structured datasets for LLM training

## Prerequisites

- Python 3.7+
- Google Generative AI API key (Gemini API)
- Required Python libraries (see Installation)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd pre-process_clean_dataset_text-only
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Obtain a Gemini API key from Google AI Studio (https://makersuite.google.com/)

## Usage Guide

### 1. Book Processor

The main processing engine that handles book processing:

```bash
python book_processor.py --root-dir "/path/to/books" \
                         --output-dir "./processed" \
                         --api-key "YOUR_API_KEY" \
                         --batch-size 5 \
                         --workers 2 \
                         --dataset-path "dataset.jsonl"
```

**Available Options:**
- `--root-dir`: Root directory containing book folders (required)
- `--output-dir`: Output directory for processed files (required)
- `--api-key`: Gemini API key (required)
- `--batch-size`: Number of books to process in a batch (default: 5)
- `--workers`: Number of parallel workers (default: 2)
- `--dataset-path`: Path for the combined dataset (default: "combined_dataset.jsonl")
- `--scan-only`: Only scan directories without processing
- `--stats`: Show processing statistics

### 2. Batch Manager

A tool for managing and monitoring the processing pipeline:

```bash
python batch-manager.py --db-path "book_processing.db" dashboard --refresh 5
```

**Available Commands:**
- `dashboard`: Display live processing dashboard
- `retry`: Reset failed books for retry
- `export`: Export book list to JSON
- `details`: Show details for a specific book
- `prioritize`: Prioritize specific books

**Examples:**
```bash
# Display live dashboard with 5-second refresh interval
python batch-manager.py dashboard --refresh 5

# Retry all failed books
python batch-manager.py retry

# Export book list to JSON
python batch-manager.py export --output books_export.json

# Show details for a specific book
python batch-manager.py details book_id_123

# Prioritize specific books
python batch-manager.py prioritize book_id_123 book_id_456
```

### 3. Chunk Viewer

A text-based UI for visualizing processing in real-time:

```bash
python chunk-viewer-tui.py --db-path "book_processing.db" --refresh 0.5
```

**Available Options:**
- `--db-path`: Path to the database file (default: "book_processing.db")
- `--refresh`: Refresh interval in seconds (default: 0.5)
- `--create-test-data`: Create test data for demo purposes

**Controls:**
- `N`: Next Chunk
- `P`: Previous Chunk
- `M`: Toggle Mode (Latest/Navigate)
- `O↑/↓`: Scroll Original Text
- `R↑/↓`: Scroll Processed Text
- `Q`: Quit

### 4. Output Analyzer

A tool for analyzing and visualizing the quality of processed text:

```bash
python output_analyzer.py --processed-file "book_name_processed.md" \
                          --dataset-file "dataset.jsonl" \
                          --output-dir "analysis" \
                          --save-json
```

**Available Options:**
- `--processed-file`: Path to processed markdown file (required)
- `--dataset-file`: Path to JSONL dataset file
- `--output-dir`: Directory to save visualizations (default: "analysis")
- `--save-json`: Save analysis as JSON

### 5. Simple Dashboard

A lightweight terminal-based dashboard for processing status:

```bash
python simple_dashboard.py --db-path "book_processing.db" --refresh 5
```

**Available Options:**
- `--db-path`: Path to the database file (default: "book_processing.db")
- `--refresh`: Refresh interval in seconds (default: 5)

## Processing Pipeline Details

### Directory Structure

The expected directory structure for book processing:

```
root-dir/
├── Book1/
│   ├── part_1.txt
│   ├── part_2.txt
│   └── part_3.txt
├── Book2/
│   ├── part_1.txt
│   └── part_2.txt
└── Book3/
    └── book3.txt
```

### Processing Workflow

1. **Scanning**: The tool scans the root directory for book folders
2. **Merging**: Each book's files are merged in chronological order
3. **Chunking**: Large texts are split into manageable chunks
4. **Processing**: Chunks are processed using Gemini API
5. **Cleaning**: Specialized cleaning rules are applied to the text
6. **Metadata Extraction**: Key information is extracted from the text
7. **Dataset Creation**: Processed text is converted to a JSONL dataset

### Database Schema

The processing pipeline uses an SQLite database with the following tables:

- `books`: Information about each book and its processing status
- `files`: Details about individual files within books
- `api_usage`: Tracking of API calls and their success/failure
- `chunk_processing`: Detailed tracking of chunk processing

## Text Processing Details

The text processing pipeline includes:

- Removal of image references, page numbers, and formatting artifacts
- Fixing and formatting Sanskrit verses and translations
- Formatting chapter headings and sections with markdown
- Preserving diacritical marks in Sanskrit terms
- Cleaning up table structures with proper markdown
- Name replacements for specific entities
- Removal of contact information, book metadata, and numerical statistics

## Monitoring and Analysis

The suite provides multiple tools for monitoring and analysis:

- **Live Dashboard**: Real-time processing status and statistics
- **Chunk Viewer**: Live visualization of text transformation
- **Output Analyzer**: Statistical analysis and quality visualization
- **Batch Manager**: Detailed control and monitoring of processing

## Performance Considerations

- Processing time depends on the book size, number of chunks, and Gemini API response time
- Parallel processing can be adjusted with the `--workers` parameter
- Batch size can be optimized with the `--batch-size` parameter
- The database allows for resuming processing after interruptions
