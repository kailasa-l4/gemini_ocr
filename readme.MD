# Gemini API Processing Toolkit

This repository contains a collection of powerful tools that leverage Google's Gemini API for various document processing, OCR, and text preparation tasks. Each project demonstrates different capabilities of the Gemini models for handling complex document processing workflows.

## Project Overview

The repository contains three specialized projects, each focusing on a different aspect of document processing:

### 1. Application Form Digitalization OCR

**Directory:** `application_form_digitalization_ocr/`

A tool designed to extract and digitize information from Hindi application forms, including automatic field detection, Hindi-to-English translation, and data validation. Perfect for organizations needing to process large volumes of form images.

**Key Features:**
- Extract and translate form fields from Hindi to English
- Process multiple forms in parallel with multi-threading
- Validate phone numbers, emails, and dates
- Export structured data to CSV format

### 2. PDF OCR Book Processor

**Directory:** `gemini_ocr_pdf_image/`

A specialized tool for extracting text from scanned PDF books using Gemini's vision capabilities, with particular attention to preserving special characters like Sanskrit diacritical marks. Ideal for digitizing books and manuscripts.

**Key Features:**
- High-quality OCR for scanned PDFs
- Preservation of Sanskrit diacritical marks
- Text cleaning and formatting
- Dataset creation for machine learning applications

### 3. Book Processing Pipeline

**Directory:** `pre-process_clean_dataset_text-only/`

A comprehensive suite for processing large collections of books, including text cleaning, formatting, and dataset preparation for LLM training. Includes tools for monitoring, visualizing, and analyzing the processing pipeline.

**Key Features:**
- Process hundreds of books with thousands of files
- Clean and standardize text content
- Extract metadata about book content
- Real-time monitoring and visualization tools
- Generate structured datasets for LLM training

## Getting Started

### Prerequisites

All projects in this toolkit require:
- Python 3.7+
- Google Generative AI API key (Gemini API)

You can obtain a Gemini API key from [Google AI Studio](https://makersuite.google.com/).

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd gemini-api-toolkit
   ```

2. Set up a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install common dependencies:
   ```bash
   pip install google-generativeai pillow
   ```

4. For project-specific dependencies, refer to the requirements.txt file in each project directory or install all dependencies with:
   ```bash
   pip install -r application_form_digitalization_ocr/requirements.txt
   pip install -r gemini_ocr_pdf_image/requirements.txt
   pip install -r pre-process_clean_dataset_text-only/requirements.txt
   ```

## Usage

Each project has its own specific usage instructions. Please refer to the README files within each project directory for detailed guidance:

- For form digitalization: See `application_form_digitalization_ocr/README.md`
- For PDF OCR: See `gemini_ocr_pdf_image/README.md`
- For the book processing pipeline: See `pre-process_clean_dataset_text-only/README.md`

## General Workflow

While each project has its specialized focus, they follow a similar general workflow:

1. **Input Processing**: Load and prepare documents (forms, PDFs, or text files)
2. **AI Processing**: Send content to Gemini API for intelligent processing
3. **Post-Processing**: Clean, format, and validate the returned data
4. **Output Generation**: Save processed data in structured formats

## API Usage Considerations

All projects in this toolkit use the Gemini API and are subject to its usage limits and pricing:

- Monitor your API usage to avoid unexpected costs
- Implement appropriate rate limiting for large batch operations
- Consider the token usage of different Gemini models