# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an enhanced OCR (Optical Character Recognition) tool that uses Google's Gemini 2.5 Flash model to extract text from PDF files and images with advanced legibility detection and semantic validation.

### Key Features
- **Optimized pre-assessment**: Combined visual legibility + semantic quality prediction in single API call
- **Multi-format support**: PDF files, single images, and directories of images
- **Resume functionality**: Progress tracking with CSV files to resume interrupted processing
- **Structured output**: JSON schemas for consistent results, conditional markdown output files
- **Quality thresholds**: Configurable legibility and semantic thresholds
- **Reduced API consumption**: 2 calls per image instead of 3 (33% reduction)
- **Smart file generation**: MD files only created for successful OCR, assessment details stored in CSV

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
   - `CombinedAssessmentResult`: Combined pre-assessment results with comprehensive quality metrics
   - `LegibilityResult`: Legibility assessment results (legacy)
   - `OCRResult`: Text extraction results  
   - `SemanticResult`: Semantic validation results (legacy)
   - `PageProgress`/`ImageProgress`: Enhanced progress tracking with detailed assessment fields

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
    ├── {name}_processed.md     # Final combined output (successful OCR only)
    ├── {name}_progress.csv     # Comprehensive progress with assessment details
    └── page_*.md or *.md       # Individual files (successful OCR only)
```

### CSV Progress Fields
The progress CSV files now contain comprehensive assessment details:
- **Basic Fields**: file_path, status, legibility_score, semantic_score, ocr_confidence, processing_time, error_message, timestamp
- **Assessment Details**: text_clarity, image_quality, ocr_prediction, semantic_prediction, visible_text_sample, language_detected, issues_found

**Status Values**:
- `completed`: Successful OCR processing (MD file created)
- `illegible`: Failed legibility threshold (no MD file, details in CSV)
- `semantically_invalid`: Failed semantic threshold (no MD file, details in CSV)
- `error`: Processing error (MD file with error message)

## Development Commands

### Setup
```bash
# Install dependencies (uses uv package manager)
uv sync

# Configure environment (copy .env.example to .env and edit)
cp .env.example .env
# Edit .env file with your settings

# Run the OCR processor
python ocr_book_processor.py --help
```

### Configuration Options

#### Environment Variables (.env file)
Create a `.env` file in the project root with your configuration:
```bash
# Required
GEMINI_API_KEY=your_gemini_api_key_here

# Optional processing parameters
LEGIBILITY_THRESHOLD=0.5
SEMANTIC_THRESHOLD=0.6
THINKING_BUDGET=2000
DPI=300

# Optional paths (can be overridden by CLI)
OUTPUT_DIR=./output
# INPUT_FILE=path/to/file.pdf
# INPUT_DIR=path/to/images

# Optional flags
SKIP_TEXT_CLEANING=false

# Optional thinking configuration
ENABLE_THINKING_ASSESSMENT=true
ENABLE_THINKING_OCR=false
```

#### Usage Examples
```bash
# With .env configuration (minimal command)
python ocr_book_processor.py --input-file book.pdf

# Override .env settings with CLI arguments
python ocr_book_processor.py --input-file document.png --legibility-threshold 0.7

# Traditional CLI-only approach (still supported)
python ocr_book_processor.py --input-file book.pdf --output-dir ./output --api-key YOUR_KEY
```

### Key Parameters
- `--legibility-threshold`: Minimum score (0-1) for visual legibility assessment (default: 0.5)
- `--semantic-threshold`: Minimum score (0-1) for semantic meaningfulness (default: 0.6)
- `--thinking-budget`: Token budget for Gemini's reasoning process (default: 2000)
- `--enable-thinking-assessment`: Enable thinking for assessment phase (default: true)
- `--enable-thinking-ocr`: Enable thinking for OCR extraction phase (default: false)
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
2. **Combined Pre-Assessment**: Single AI call with thinking-enabled evaluation:
   - Visual legibility (can text be read?)
   - Expected semantic quality (does visible text look meaningful?)
   - Comprehensive quality metrics and predictions
   - Decision: proceed to OCR only if both thresholds met
3. **Conditional OCR Extraction**: Text extraction (only if pre-assessment passed)
4. **Text Cleaning**: Post-processing and formatting (successful cases only)
5. **Smart Output Generation**: 
   - **Successful OCR**: Create MD files with extracted text
   - **Failed Assessment**: Store detailed assessment in CSV only (no MD file)
6. **Comprehensive Progress Tracking**: Enhanced CSV files with full assessment details

**Benefits vs. Previous 3-Step Process:**
- Reduced API calls from 3 → 2 per successful image
- Reduced API calls from 2-3 → 1 per failed image
- Lower latency and costs
- **Enhanced data storage**: All assessment details preserved in CSV
- **Cleaner output**: MD files only for meaningful OCR results

## Assessment Quality Metrics

The combined pre-assessment provides detailed quality analysis stored in CSV files:

### Core Metrics
- **legibility_score**: Visual clarity score (0-1)
- **expected_semantic_quality**: Predicted meaningfulness of OCR output (0-1)
- **text_clarity**: Categorical assessment (poor/fair/good/excellent)
- **image_quality**: Overall image quality (poor/fair/good/excellent)

### Prediction Metrics
- **ocr_prediction**: Expected OCR quality (excellent/good/fair/poor/unusable)
- **semantic_prediction**: Expected text meaningfulness (meaningful_text/partial_meaning/fragmented/mostly_gibberish/unusable)
- **visible_text_sample**: Sample of clearly readable text from image
- **language_detected**: Primary language/script identified
- **issues_found**: Specific quality issues affecting OCR (comma-separated)

### Thinking-Enabled Processing
The system can use Gemini's thinking capability for detailed analysis in two phases:

**Assessment Phase (ENABLE_THINKING_ASSESSMENT=true by default):**
1. Visual examination of text clarity and image quality
2. Character-level legibility assessment
3. Semantic quality prediction based on visible content
4. Language and script analysis
5. Final decision on OCR viability

**OCR Phase (ENABLE_THINKING_OCR=false by default):**
- Can be enabled for complex documents requiring detailed reasoning during text extraction
- Useful for damaged documents, complex layouts, or multilingual content
- Increases processing time but may improve accuracy for challenging content

## Text Processing Rules

The tool includes specific text replacement rules:
- Replace "Paramahamsa Nithyananda" → "THE SUPREME PONTIFF OF HINDUISM BHAGAWAN SRI NITHYANANDA PARAMASHIVAM"
- Replace "India" → "Bharat", "Indian" → "Hindu"
- Remove contact information, book metadata, and statistics
- Preserve Sanskrit diacritical marks and formatting