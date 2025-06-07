# OCR Modules

This directory contains the modularized components of the Enhanced OCR processor with Gemini 2.5 Flash.

## Module Structure

### Core Modules

- **`models.py`** - Data classes and structures
- **`schemas.py`** - JSON schemas for Gemini API responses  
- **`utils.py`** - Utility functions for file operations
- **`progress_manager.py`** - CSV progress tracking and management

### Processing Components

- **`ocr_engine.py`** - Core OCR engine with Gemini API integration
- **`processors.py`** - Document processors (PDF, single image, directory)
- **`ocr_processor_modular.py`** - Main orchestration class

### Entry Points

- **`main.py`** - Standalone entry point (alternative to root file)
- **`__init__.py`** - Package initialization and exports

## Usage

### From the main directory:
```python
from ocr_modules import GeminiAdvancedOCR
ocr = GeminiAdvancedOCR(api_key="your_key")
```

### Individual components:
```python
from ocr_modules.ocr_engine import GeminiOCREngine
from ocr_modules.processors import PDFProcessor
from ocr_modules.models import PageProgress
```

## Benefits

- **Modularity**: Each file has a single responsibility
- **Maintainability**: Easy to understand and modify individual components
- **Testability**: Components can be tested in isolation
- **Reusability**: Individual modules can be imported as needed
- **Clean Imports**: Organized package structure with proper __init__.py