# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an enhanced OCR (Optical Character Recognition) tool that uses Google's Gemini 2.5 Flash model to extract text from PDF files and images with advanced legibility detection and semantic validation.

### Key Features
- **Optimized pre-assessment**: Combined visual legibility + semantic quality prediction in single API call
- **Multi-format support**: PDF files, single images, and directories of images
- **Resume functionality**: Progress tracking with CSV files to resume interrupted processing
- **Structured output**: JSON schemas for consistent results, markdown output files
- **Quality thresholds**: Configurable legibility and semantic thresholds
- **Reduced API consumption**: 2 calls per image instead of 3 (33% reduction)

### Supported Formats
Images: JPG, JPEG, PNG, BMP, TIFF, TIF, WEBP

## Architecture

### Core Components

1. **GeminiAdvancedOCR** (`ocr_book_processor.py`): Main processing class
   - `combined_pre_assessment()`: Evaluates both legibility and semantic quality in one call
   - `extract_text()`: Performs actual OCR extraction
   - `clean_and_process_text()`: Post-processes extracted text
   - Legacy methods: `assess_legibility()`, `validate_semantic_meaning()` (kept for compatibility)

2. **Data Classes**: 
   - `CombinedAssessmentResult`: Combined pre-assessment results (NEW)
   - `LegibilityResult`: Legibility assessment results (legacy)
   - `OCRResult`: Text extraction results  
   - `SemanticResult`: Semantic validation results (legacy)
   - `PageProgress`/`ImageProgress`: Progress tracking

3. **Processing Methods**:
   - `process_pdf()`: Handle PDF files with page-by-page processing
   - `process_single_image()`: Process individual image files
   - `process_images()`: Batch process directories of images

### Output Structure
```
output/
├── {filename}_image/           # For single images
├── {filename}/                 # For PDFs
└── {dirname}_images/           # For image directories
    ├── {name}_processed.md     # Final combined output
    ├── {name}_progress.csv     # Processing progress
    └── page_*.md or *.md       # Individual processed files
```

## Development Commands

### Setup
```bash
# Install dependencies (uses uv package manager)
uv sync

# Run the OCR processor
python ocr_book_processor.py --help
```

### Usage Examples
```bash
# Process a PDF
python ocr_book_processor.py --input-file book.pdf --output-dir ./output --api-key YOUR_KEY

# Process single image with custom thresholds
python ocr_book_processor.py --input-file document.png --output-dir ./output --api-key YOUR_KEY --legibility-threshold 0.6 --semantic-threshold 0.7

# Process directory of images
python ocr_book_processor.py --input-dir ./images --output-dir ./output --api-key YOUR_KEY
```

### Key Parameters
- `--legibility-threshold`: Minimum score (0-1) for visual legibility assessment
- `--semantic-threshold`: Minimum score (0-1) for semantic meaningfulness
- `--thinking-budget`: Token budget for Gemini's reasoning process
- `--dpi`: Resolution for PDF page rendering (default: 300)

## Configuration

### Environment Requirements
- Google Gemini API key required
- Uses `gemini-2.5-flash-preview-05-20` model
- Requires Python 3.12+

### Dependencies
- `google-genai`: Google's Gen AI SDK for Gemini
- `pymupdf`: PDF processing (fitz)
- `pillow`: Image processing
- `tqdm`: Progress bars

## Processing Pipeline

**Optimized 2-Step Process:**

1. **Input Validation**: Check file formats and existence
2. **Combined Pre-Assessment**: Single AI call evaluates:
   - Visual legibility (can text be read?)
   - Expected semantic quality (does visible text look meaningful?)
   - Decision: proceed to OCR only if both thresholds met
3. **OCR Extraction**: Text extraction (only if pre-assessment passed)
4. **Text Cleaning**: Post-processing and formatting
5. **Output Generation**: Markdown files with structured results
6. **Progress Tracking**: CSV files for resume capability

**Benefits vs. Previous 3-Step Process:**
- Reduced API calls from 3 → 2 per successful image
- Reduced API calls from 2-3 → 1 per failed image
- Lower latency and costs

## Text Processing Rules

The tool includes specific text replacement rules:
- Replace "Paramahamsa Nithyananda" → "THE SUPREME PONTIFF OF HINDUISM BHAGAWAN SRI NITHYANANDA PARAMASHIVAM"
- Replace "India" → "Bharat", "Indian" → "Hindu"
- Remove contact information, book metadata, and statistics
- Preserve Sanskrit diacritical marks and formatting