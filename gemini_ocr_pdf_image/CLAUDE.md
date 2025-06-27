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
   - `extract_text()`: Performs actual OCR extraction with markdown formatting
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

# Test database connection (optional)
python test_db_integration.py

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

# Optional database logging configuration
ENABLE_DATABASE_LOGGING=true
DATABASE_URL=postgresql://ocr:rmtetn5qek6zikol@157.180.15.165:1111/db
DATABASE_CONNECTION_TIMEOUT=30
DATABASE_RETRY_ATTEMPTS=3

# Optional logging configuration
LOGS_DIR=./logs
CSV_LOGS_DIR=./logs

# Optional paths (can be overridden by CLI)
OUTPUT_DIR=./output
# INPUT_FILE=path/to/file.pdf
# INPUT_DIR=path/to/images

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
- `psycopg2-binary`: PostgreSQL database connectivity (optional)

## Database Logging

The OCR processor includes comprehensive PostgreSQL database logging capabilities:

### Features
- **Session Tracking**: Each OCR run gets a unique session ID with metadata
- **Processing Logs**: Detailed logs for every page/image processed
- **Error Logging**: System errors and exceptions with stack traces
- **Multi-Session Support**: Handle concurrent OCR processes from different locations
- **Fail-Fast Validation**: OCR won't start if database is enabled but unavailable
- **Dual Storage**: Both local CSV files (in `/logs` folder) AND database logging

### Database Schema
```sql
-- OCR processing sessions (multiple sessions can run simultaneously)
CREATE TABLE ocr_sessions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(50) UNIQUE,
    hostname VARCHAR(100),
    start_time TIMESTAMP DEFAULT NOW(),
    end_time TIMESTAMP,
    input_path VARCHAR(500),
    input_type VARCHAR(20), -- 'pdf', 'image', 'directory'
    output_path VARCHAR(500),
    status VARCHAR(20), -- 'running', 'completed', 'failed', 'interrupted'
    total_files INTEGER,
    completed_files INTEGER,
    failed_files INTEGER,
    configuration JSONB
);

-- Individual file processing logs with comprehensive assessment details
CREATE TABLE processing_logs (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(50) REFERENCES ocr_sessions(session_id),
    file_path VARCHAR(500),
    page_number INTEGER,
    processing_start TIMESTAMP DEFAULT NOW(),
    processing_end TIMESTAMP,
    status VARCHAR(30), -- 'completed', 'illegible', 'semantically_invalid', 'error'
    legibility_score DECIMAL(3,2),
    semantic_score DECIMAL(3,2),
    ocr_confidence DECIMAL(3,2),
    processing_time DECIMAL(10,6),
    text_clarity VARCHAR(20),
    image_quality VARCHAR(20),
    ocr_prediction VARCHAR(30),
    semantic_prediction VARCHAR(30),
    visible_text_sample TEXT,
    language_detected VARCHAR(50),
    issues_found TEXT,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- System errors and exceptions
CREATE TABLE error_logs (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(50),
    error_type VARCHAR(100),
    error_message TEXT,
    stack_trace TEXT,
    file_path VARCHAR(500),
    function_name VARCHAR(100),
    line_number INTEGER,
    severity VARCHAR(20), -- 'low', 'medium', 'high', 'critical'
    hostname VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW()
);
```

### Configuration
Set `ENABLE_DATABASE_LOGGING=true` in your `.env` file and provide the database URL:
```bash
ENABLE_DATABASE_LOGGING=true
DATABASE_URL=postgresql://ocr:rmtetn5qek6zikol@157.180.15.165:1111/db
```

### Usage
```bash
# With database logging enabled (via .env)
python ocr_book_processor.py --input-file document.pdf

# Disable database logging for specific run
python ocr_book_processor.py --input-file document.pdf --disable-database-logging

# Test database connection
python test_db_integration.py
```

## Processing Pipeline

**Optimized 2-Step Process:**

1. **Input Validation**: Check file formats and existence
2. **Combined Pre-Assessment**: Single AI call with thinking-enabled evaluation:
   - Visual legibility (can text be read?)
   - Expected semantic quality (does visible text look meaningful?)
   - Comprehensive quality metrics and predictions
   - Decision: proceed to OCR only if both thresholds met
3. **Conditional OCR Extraction with Markdown Formatting**: Text extraction with markdown formatting applied during extraction (only if pre-assessment passed)
4. **Smart Output Generation**: 
   - **Successful OCR**: Create MD files with properly formatted markdown text
   - **Failed Assessment**: Store detailed assessment in CSV only (no MD file)
5. **Comprehensive Progress Tracking**: Enhanced CSV files with full assessment details

**Benefits vs. Previous Process:**
- **Single-step formatting**: Markdown formatting applied during OCR extraction, not as separate step
- **Reduced API calls**: No separate text cleaning API call needed
- **Better formatting**: Gemini can see visual structure and apply appropriate markdown formatting
- **Cleaner architecture**: No post-processing step required
- **Enhanced data storage**: All assessment details preserved in CSV
- **Cleaner output**: MD files only for meaningful OCR results with proper markdown formatting

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

## Markdown Formatting

The OCR extraction now automatically applies proper markdown formatting during text extraction:

### Supported Markdown Elements with Strict Consistency Rules
- **Headings**: 
  - `#` for main document title only
  - `## Page X` for page identifiers
  - `##` for major sections (CONTENTS, CHAPTER titles)
  - `###` for subsections within chapters
  - Never multiple `#` headings for parts of the same title
- **Lists with Consistent Formatting**:
  - **Table of Contents**: Always bullet lists (*) with page numbers
  - **Chapter Content**: Always bullet points (-) with italic emphasis for topic titles
  - **Feature Lists**: Consistent bullet points throughout
  - **Never mix** different list types within the same section type
- **Emphasis**: Consistent `*italic*` for topic titles in lists, sparing use of `**bold**`
- **Spacing**: Standardized blank lines between sections for uniform appearance

### 2-Column Layout Handling
The OCR system now properly handles 2-column page layouts:
- **Reading Order**: Left column first (top to bottom), then right column (top to bottom)
- **Logical Sequence**: Maintains left-to-right, top-to-bottom reading flow in output
- **Structure Preservation**: Content from both columns is merged in correct reading order

### Enhanced Formatting Rules
The system now follows strict formatting patterns:

**Contents Section Format:**
```markdown
# CONTENTS

## I SECTION NAME
* Topic Name - Page Number
* Another Topic - Page Number

## II NEXT SECTION  
* Topic Name - Page Number
* Another Topic - Page Number
```

**Chapter Lists Format:**
```markdown
## CHAPTER NAME
- *Topic Title* Page Number
- *Another Topic* Page Number
- *Third Topic* Page Number
```

### Benefits of Strict Formatting
- **Consistent visual hierarchy**: Clear, predictable heading structure
- **Uniform list formatting**: No mixing of list types within sections
- **Professional appearance**: Standardized spacing and emphasis patterns
- **Better readability**: Consistent formatting makes content easier to navigate
- **Quality assurance**: Built-in formatting validation before output

## Text Processing Rules

The tool includes specific text replacement rules applied during OCR extraction:
- Replace "Paramahamsa Nithyananda" → "THE SUPREME PONTIFF OF HINDUISM BHAGAWAN SRI NITHYANANDA PARAMASHIVAM"
- Replace "India" → "Bharat", "Indian" → "Hindu"
- Remove contact information, book metadata, and statistics during extraction
- Preserve Sanskrit diacritical marks and formatting