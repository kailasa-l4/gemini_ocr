import os
import re
import json
import time
import argparse
import tempfile
from tqdm import tqdm
import google.generativeai as genai
from PIL import Image
import fitz  # PyMuPDF

class GeminiVisionOCR:
    """Use Gemini Vision capabilities to OCR scanned PDFs."""
    
    def __init__(self, api_key, model_name="gemini-2.0-flash-001"):
        """Initialize the OCR processor."""
        self.api_key = api_key
        self.model_name = model_name
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
    
    def process_pdf(self, pdf_path, output_dir, start_page=1, end_page=None, 
                    dpi=300, batch_size=1):
        """
        Process a PDF file with Gemini Vision OCR.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save output
            start_page: First page to process (1-based)
            end_page: Last page to process (inclusive)
            dpi: DPI for rendering PDF pages
            batch_size: Number of pages to process in each API call
            
        Returns:
            Path to the processed file
        """
        # Create output directory
        book_name = os.path.splitext(os.path.basename(pdf_path))[0]
        book_output_dir = os.path.join(output_dir, book_name)
        os.makedirs(book_output_dir, exist_ok=True)
        
        # Path for OCR text
        ocr_file_path = os.path.join(book_output_dir, f"{book_name}_ocr.txt")
        processed_file_path = os.path.join(book_output_dir, f"{book_name}_processed.md")
        
        # Check if already processed
        if os.path.exists(processed_file_path):
            print(f"File already processed: {processed_file_path}")
            return processed_file_path
        
        # Open the PDF
        print(f"Opening PDF: {pdf_path}")
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        print(f"PDF has {total_pages} pages")
        
        # Adjust page range
        if start_page < 1:
            start_page = 1
        
        if end_page is None or end_page > total_pages:
            end_page = total_pages
        
        pages_to_process = range(start_page-1, end_page)  # 0-based index
        num_pages = len(pages_to_process)
        
        print(f"Processing pages {start_page} to {end_page} ({num_pages} pages)")
        
        # Process each page
        all_text = {}
        with open(ocr_file_path, 'w', encoding='utf-8') as ocr_file:
            # Add book title
            ocr_file.write(f"# {book_name}\n\n")
            
            # Process batches of pages
            with tqdm(total=num_pages, desc="OCR Progress", unit="page") as pbar:
                for i in pages_to_process:
                    page_num = i + 1  # Convert to 1-based index for display
                    
                    try:
                        # Render the page as an image
                        pbar.set_description(f"Rendering page {page_num}")
                        
                        # Get the page
                        page = doc[i]
                        
                        # Render to an image (in memory)
                        pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
                        
                        # Convert to PIL Image
                        mode = "RGBA" if pix.alpha else "RGB"
                        img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
                        
                        # Perform OCR with Gemini Vision
                        pbar.set_description(f"OCR page {page_num}")
                        ocr_text = self.ocr_image_with_gemini(img, page_num)
                        
                        # Save OCR result
                        all_text[page_num] = ocr_text
                        ocr_file.write(f"\n\n## Page {page_num}\n\n")
                        ocr_file.write(ocr_text)
                        ocr_file.flush()  # Ensure file is written
                        
                        # Update progress
                        pbar.update(1)
                        pbar.set_description(f"Page {page_num}/{end_page}")
                        
                        # Sleep to avoid API limits
                        time.sleep(1)
                        
                    except Exception as e:
                        print(f"\nError processing page {page_num}: {str(e)}")
                        error_msg = f"[Error processing page {page_num}: {str(e)}]"
                        all_text[page_num] = error_msg
                        ocr_file.write(f"\n\n## Page {page_num}\n\n")
                        ocr_file.write(error_msg)
                        ocr_file.flush()
                        
                        # Update progress
                        pbar.update(1)
        
        # Close the document
        doc.close()
        
        print(f"\nOCR completed. Text saved to: {ocr_file_path}")
        
        # Clean and process the OCR text with Gemini
        print("\nProcessing OCR text...")
        
        # Read the OCR file
        with open(ocr_file_path, 'r', encoding='utf-8') as f:
            ocr_text = f.read()
        
        # Split into batches based on page markers
        page_markers = re.finditer(r'\n\n## Page \d+\n\n', ocr_text)
        page_positions = [0] + [m.start() for m in page_markers]
        if page_positions[-1] < len(ocr_text):
            page_positions.append(len(ocr_text))
        
        # Group into reasonable sized chunks
        chunks = []
        current_chunk = ""
        
        for i in range(len(page_positions) - 1):
            page_chunk = ocr_text[page_positions[i]:page_positions[i+1]]
            
            # If adding this chunk exceeds our limit, start a new chunk
            if len(current_chunk) + len(page_chunk) > 10000 and current_chunk:
                chunks.append(current_chunk)
                current_chunk = page_chunk
            else:
                current_chunk += page_chunk
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        # Process each chunk
        processed_chunks = []
        with tqdm(total=len(chunks), desc="Processing text", unit="chunk") as pbar:
            for i, chunk in enumerate(chunks):
                try:
                    # Process with Gemini
                    pbar.set_description(f"Chunk {i+1}/{len(chunks)}")
                    processed_chunk = self.process_text_with_gemini(chunk)
                    processed_chunks.append(processed_chunk)
                    
                except Exception as e:
                    print(f"\nError processing chunk {i+1}: {str(e)}")
                    processed_chunks.append(chunk)  # Use original on error
                
                # Update progress
                pbar.update(1)
                
                # Sleep between chunks
                time.sleep(1)
        
        # Combine processed chunks
        processed_text = "".join(processed_chunks)
        
        # Save processed text
        with open(processed_file_path, 'w', encoding='utf-8') as f:
            f.write(processed_text)
        
        print(f"\nProcessing completed. Saved to: {processed_file_path}")
        return processed_file_path
    
    def ocr_image_with_gemini(self, img, page_num):
        """
        Use Gemini Vision to extract text from an image.
        
        Args:
            img: PIL Image
            page_num: Page number for logging
            
        Returns:
            Extracted text
        """
        # Create the prompt for OCR
        prompt = """
        Extract ALL text from this scanned page image. This is a page from a book that may contain Sanskrit terms with diacritical marks.
        
        Guidelines:
        1. Capture ALL text in the image including headings, paragraphs, footnotes
        2. Preserve the original text structure and formatting
        3. Maintain paragraph breaks and line structure
        4. Preserve all diacritical marks in Sanskrit terms (ā, ī, ū, ṛ, ṝ, ḷ, ḹ, ṃ, ḥ, etc.)
        5. Include any visible verse numbers or section markers
        6. Format tables accurately preserving rows and columns
        7. Ignore watermarks and purely decorative elements
        8. Return ONLY the extracted text, no explanations or metadata
        """
        
        # Call Gemini Vision API
        response = self.model.generate_content([prompt, img])
        return response.text.strip()
    
    def process_text_with_gemini(self, text):
        """Process OCR text with Gemini to clean and format it."""
        prompt = f"""
        You're cleaning OCRed text from a spiritual book for LLM training. Clean and structure this text while preserving its meaning:
        
        1. Remove image references, page numbers, and formatting artifacts
        2. Fix and properly format Sanskrit verses and translations
        3. Format chapter headings and sections with markdown
        4. Preserve diacritical marks in Sanskrit terms
        5. Clean up any table structures with proper markdown
        6. Any reference of {{"Paramahamsa Nithyananda", "Nithyananda"}} make it as "THE SUPREME PONTIFF OF HINDUISM BHAGAWAN SRI NITHYANANDA PARAMASHIVAM", if it is already mentioned as THE "SUPREME PONTIFF OF HINDUISM BHAGAWAN SRI NITHYANANDA PARAMASHIVAM" then do not change.
        7. Replace "India" with "Bharat" and "Indian" with Hindu
        8. Remove ALL contact information (phone numbers, emails, websites, social media handles)
        9. Remove ALL book metadata (ISBN numbers, copyright notices, publication dates, pricing, publisher information)
        10. Remove the entire index section of the book and any unnecessary line breaks in the text
        11. Remove specific numerical statistics about humanitarian activities (e.g., "10,000 meals served", "1,000 people healed") - keep only general descriptions without numbers
        12. Give the response according to the language as in original text, do not translate anything to english or any other language.
        13. Return ONLY the cleaned text, no explanations

        Text to clean:
        ```
        {text}
        ```
        """
        
        response = self.model.generate_content(prompt)
        return response.text.strip()


def create_dataset(processed_files, output_file, min_length=100):
    """
    Create a JSONL dataset from processed files.
    
    Args:
        processed_files: List of processed markdown files
        output_file: Path to output JSONL file
        min_length: Minimum word count for examples
    """
    print(f"\nCreating dataset from {len(processed_files)} processed files...")
    
    examples = []
    
    # Process each file
    with tqdm(total=len(processed_files), desc="Creating dataset", unit="file") as pbar:
        for file_path in processed_files:
            try:
                # Skip non-existent files
                if not os.path.exists(file_path):
                    pbar.update(1)
                    continue
                
                book_name = os.path.splitext(os.path.basename(file_path))[0].replace("_processed", "")
                
                # Read the file
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                # Split into sections (by headings)
                sections = re.split(r'(?=^# |\n# |\n## |\n### )', text, flags=re.MULTILINE)
                
                # Create real sections by removing page breaks
                content_sections = []
                current_section = ""
                
                for section in sections:
                    # Skip page markers and empty sections
                    if not section.strip() or re.match(r'^#+\s*Page\s+\d+\s*$', section.strip()):
                        continue
                    
                    # Remove internal page markers
                    section = re.sub(r'\n\n## Page \d+\n\n', '\n\n', section)
                    
                    # If it's a chapter heading, start a new section
                    if re.match(r'^#+\s', section.strip()):
                        if current_section:
                            content_sections.append(current_section.strip())
                        current_section = section
                    else:
                        current_section += section
                
                # Add the last section
                if current_section:
                    content_sections.append(current_section.strip())
                
                # Create examples from content sections
                file_examples = []
                for section in content_sections:
                    # Check minimum length
                    words = section.split()
                    if len(words) >= min_length:
                        example = {
                            "text": section,
                            "source": book_name
                        }
                        file_examples.append(example)
                
                examples.extend(file_examples)
                pbar.set_postfix(examples=len(examples))
                pbar.update(1)
                
            except Exception as e:
                print(f"\nError processing file {file_path}: {str(e)}")
                pbar.update(1)
    
    # Write dataset to JSONL file
    print(f"\nWriting {len(examples)} examples to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"Dataset created with {len(examples)} examples")
    return len(examples)


def find_pdf_files(input_dir):
    """Find all PDF files in a directory (including subdirectories)."""
    pdf_files = []
    
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                pdf_files.append(pdf_path)
    
    return pdf_files


def main():
    parser = argparse.ArgumentParser(description="OCR PDFs using Gemini Vision")
    parser.add_argument("--input-dir", help="Input directory containing PDF files")
    parser.add_argument("--input-file", help="Single PDF file to process")
    parser.add_argument("--output-dir", required=True, help="Output directory for OCR results")
    parser.add_argument("--api-key", required=True, help="Gemini API key")
    parser.add_argument("--model", default="gemini-2.0-flash-001", help="Gemini model name")
    parser.add_argument("--start-page", type=int, default=1, help="Start page (1-based index)")
    parser.add_argument("--end-page", type=int, default=None, help="End page (inclusive)")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for rendering PDF pages")
    parser.add_argument("--dataset", help="Create dataset and save to this path")
    parser.add_argument("--min-length", type=int, default=100, help="Minimum word count for dataset examples")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.input_file and not args.input_dir:
        print("Error: Either --input-file or --input-dir must be specified")
        return
    
    # Create OCR processor
    ocr = GeminiVisionOCR(args.api_key, args.model)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    processed_files = []
    
    # Process a single file
    if args.input_file:
        if not os.path.exists(args.input_file):
            print(f"Error: File not found: {args.input_file}")
            return
        
        print(f"\n{'='*80}\nProcessing: {args.input_file}\n{'='*80}")
        processed_file = ocr.process_pdf(
            args.input_file, 
            args.output_dir,
            args.start_page,
            args.end_page,
            args.dpi
        )
        
        if processed_file:
            processed_files.append(processed_file)
    
    # Process all PDFs in directory
    elif args.input_dir:
        pdf_files = find_pdf_files(args.input_dir)
        print(f"Found {len(pdf_files)} PDF files")
        
        for pdf_file in pdf_files:
            print(f"\n{'='*80}\nProcessing: {pdf_file}\n{'='*80}")
            processed_file = ocr.process_pdf(
                pdf_file, 
                args.output_dir,
                args.start_page,
                args.end_page,
                args.dpi
            )
            
            if processed_file:
                processed_files.append(processed_file)
    
    # Create dataset if requested
    if args.dataset and processed_files:
        create_dataset(processed_files, args.dataset, args.min_length)


if __name__ == "__main__":
    main()