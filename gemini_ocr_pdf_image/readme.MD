# Gemini OCR Book Processor

A powerful tool for extracting and processing text from scanned PDFs using Google's Gemini Vision capabilities. This tool renders PDF pages as images, processes them with Gemini's vision model for OCR, cleans the extracted text, and optionally creates a structured dataset from the processed books.

## Features

- **High-quality OCR** using Google's Gemini Vision model
- **Page-by-page processing** with progress tracking
- **PDF rendering** with customizable DPI
- **Text cleaning and processing** after extraction
- **Sanskrit diacritical marks preservation**
- **Batch processing** of multiple PDFs
- **Dataset creation** for machine learning applications
- **Customizable processing pipeline**

## Prerequisites

- Python 3.7+
- Google Generative AI API key (Gemini API)
- Required Python libraries (see Installation)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd gemini_ocr_pdf_image
   ```

2. Install the required packages:
   ```bash
   pip install google-generativeai pillow pymupdf tqdm
   ```

3. Obtain a Gemini API key from Google AI Studio (https://makersuite.google.com/)

## Usage

### Basic Usage

Process a single PDF:

```bash
python ocr_book_processor.py --input-file "path/to/book.pdf" --output-dir "./output" --api-key "YOUR_API_KEY"
```

Process all PDFs in a directory:

```bash
python ocr_book_processor.py --input-dir "path/to/books" --output-dir "./output" --api-key "YOUR_API_KEY"
```

### Advanced Options

The script supports several additional options:

```bash
python ocr_book_processor.py --input-dir "path/to/books" \
                            --output-dir "./output" \
                            --api-key "YOUR_API_KEY" \
                            --model "gemini-2.0-flash-001" \
                            --start-page 10 \
                            --end-page 50 \
                            --dpi 300 \
                            --dataset "dataset.jsonl" \
                            --min-length 100
```

### Available Options

- `--input-dir`: Directory containing PDF files
- `--input-file`: Single PDF file to process
- `--output-dir`: Output directory for OCR results (required)
- `--api-key`: Gemini API key (required)
- `--model`: Gemini model name (default: "gemini-2.0-flash-001")
- `--start-page`: First page to process (1-based, default: 1)
- `--end-page`: Last page to process (inclusive, default: process all pages)
- `--dpi`: DPI for rendering PDF pages (default: 300)
- `--dataset`: Create dataset and save to this path
- `--min-length`: Minimum word count for dataset examples (default: 100)

## How It Works

The tool follows these processing steps:

1. **PDF Page Rendering**: Pages are rendered as high-quality images using PyMuPDF
2. **OCR with Gemini**: Each page image is sent to Gemini Vision API for text extraction
3. **Text Cleaning**: Extracted text is processed to remove artifacts, fix formatting, and normalize content
4. **Text Processing**: Additional cleaning is applied to prepare text for specific use cases
5. **Dataset Creation**: Optionally creates a dataset in JSONL format from the processed text

## Output Structure

For each processed PDF, the tool generates:

- **Raw OCR Text**: Extracted text with page markings
- **Processed Text**: Cleaned and formatted text in Markdown
- **Dataset Examples**: When dataset creation is enabled

Each processed book is saved in its own directory within the output folder:

```
output/
└── book_name/
    ├── book_name_ocr.txt
    └── book_name_processed.md
```

## Text Processing Details

The text processing pipeline includes:

- Removal of image references, page numbers, and formatting artifacts
- Fixing and formatting Sanskrit verses and translations
- Formatting chapter headings and sections with markdown
- Preserving diacritical marks in Sanskrit terms
- Cleaning up any table structures
- Specialized text replacements for specific content

## Dataset Creation

When enabled, the tool creates a JSONL dataset with one example per line:

```json
{"text": "Example text content...", "source": "Book Title"}
```

Dataset examples are filtered to contain at least the specified minimum number of words.

## Performance Considerations

- Processing time depends on the PDF size, number of pages, and Gemini API response time
- Higher DPI settings provide better OCR results but increase processing time
- The tool implements rate limiting to prevent API throttling

## Error Handling

The tool implements robust error handling:
- Errors during page processing are logged and processing continues with the next page
- Failed OCR attempts are marked and can be retried
- Processing errors are properly documented in the output files