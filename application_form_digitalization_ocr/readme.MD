# Hindi Application Form Digitalization Tool

This tool uses Google's Gemini AI to digitize and extract information from Hindi application forms. It automatically detects and translates form fields from Hindi to English, validates the data, and exports structured information to CSV format.

## Features

- **Automated extraction** of form data from images using Google's Gemini Vision API
- **Translation** from Hindi to English with preservation of original names and spellings
- **Data validation** for phone numbers, email addresses, and dates
- **Parallel processing** to handle multiple forms simultaneously
- **Graceful shutdown** with data preservation
- **Retry mechanism** for API failures
- **Progress tracking** with estimated completion time
- **Support for batch processing** with customizable batch sizes

## Prerequisites

- Python 3.6+
- Google Generative AI API key (Gemini API)
- Required Python libraries (see Installation)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd application_form_digitalization_ocr
   ```

2. Install the required packages:
   ```bash
   pip install google-generativeai pandas pillow
   ```

3. Obtain a Gemini API key from Google AI Studio (https://makersuite.google.com/)

## Usage

### Command Line Interface

The script can be run in different modes:

**Interactive Mode:**
```bash
python gemini_ocr.py
```

**Command Line Arguments:**
```bash
python gemini_ocr.py --api-key YOUR_API_KEY --image-dir /path/to/forms --output-csv results.csv --workers 4 --batch-size 20
```

**Process a Single Test Image:**
```bash
python gemini_ocr.py --api-key YOUR_API_KEY --test-image /path/to/image.jpg
```

### Available Options

- `--api-key`: Your Gemini API key
- `--image-dir`: Directory containing form images
- `--output-csv`: Path for output CSV file
- `--test-image`: Path to a single image for testing
- `--workers`: Number of parallel workers (2-8, default: 4)
- `--batch-size`: Number of images to process in each batch (10-50, default: 20)
- `--resume`: Resume processing from where it left off

### Example

```bash
python gemini_ocr.py --api-key "YOUR_API_KEY" --image-dir "./forms" --output-csv "./output/extracted_data.csv" --workers 4 --batch-size 20
```

## Form Field Extraction

The tool extracts the following fields from Hindi application forms:

- Application Number
- Citizenship Type (Citizenship or E-citizenship)
- Full Name (पूरा नाम)
- Nationality (राष्ट्रीयता)
- Date of Birth (जन्म तिथि)
- Place of Birth (जन्म स्थान)
- Gender (लिंग)
- Phone (फोन)
- Email (ईमेल)
- Address Lines (पता)
- Educational History (शैक्षिक इतिहास)
- Business History (व्यवसाय इतिहास)
- Participation in Kailas program (कैलास के कार्यक्रम में भाग लिया)
- Spiritual Name (आध्यात्मिक नाम)
- Parents information
- Previous Home Address (पिछले घर का पता)
- Parent's Phone
- Signature Date

## Performance Optimization

- Images are automatically resized and optimized before processing
- Multi-threading enables processing multiple forms simultaneously
- Exponential backoff retry mechanism helps handle API rate limits
- Forms are processed in batches to efficiently manage resources

## Error Handling

The tool implements robust error handling:
- Automatic retries for API failures
- Fallback data creation when extraction fails
- Detailed logging of errors and processing status
- Graceful shutdown with Ctrl+C that preserves processed data

## Output Format

The tool generates a CSV file with:
- All extracted form fields
- Processing status
- Timestamp of processing
- Original filename

## Notes

- For best results, ensure form images are clear and well-lit
- The tool works best with the standard form format shown in examples
- Processing time depends on the Gemini API response time and number of workers
- When using multiple workers, be mindful of API rate limits