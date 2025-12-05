# Data Loading Architecture - Refactored

## File Structure

### Utility Modules (in `backend/src/utils/`)

#### `pdf_processor.py`
Handles PDF text extraction and chunking logic:
- `extract_text_from_pdf(pdf_path)` - Extract text from PDF files
- `chunk_text(text, chunk_size, overlap)` - Split text into overlapping chunks
- `list_pdf_files(directory)` - List all PDF files in a directory

#### `data_utils.py`
Handles CSV/JSON loading and data filtering:
- `load_csv_file(file_path)` - Load CSV data into list of dicts
- `load_json_file(file_path)` - Load JSON data
- `find_record_by_field(data, field, value)` - Find single record by field
- `filter_records_by_field(data, field, value)` - Filter multiple records
- `get_unique_values(data, field)` - Get unique values in a field
- `merge_data_sources(csv_data, json_data)` - Merge multiple data sources

#### `__init__.py`
Exports all utilities for easy importing

### Main Data Loading Module

#### `data_ingestion/data_loading.py`
Simplified to focus on high-level operations using utilities:
- `DataLoader` - Transactional data access (patient info, visits, coverage)
- `ComplianceDocumentProcessor` - Knowledge document processing for vector DB

## Benefits

✅ **Separation of Concerns** - Utilities are reusable across the project
✅ **Cleaner Code** - Main loader focuses on business logic
✅ **Easier Testing** - Utilities can be unit tested independently
✅ **Better Maintainability** - Changes to PDF extraction logic only in one place
✅ **Reusability** - Utilities can be imported by other agents/tools

## Usage Examples

### Use Patient Data
```python
from backend.data_ingestion.data_loading import DataLoader

loader = DataLoader()
member = loader.get_member_by_id("MBR156655633")
visits = loader.get_member_visits("MBR156655633")
```

### Use Document Processing
```python
from backend.data_ingestion.data_loading import ComplianceDocumentProcessor

processor = ComplianceDocumentProcessor()
docs = processor.load_all_knowledge_documents()
```

### Use Utilities Directly
```python
from backend.src.utils import chunk_text, load_csv_file, find_record_by_field

# Chunk text for vector DB
chunks = chunk_text(text, chunk_size=500, overlap=100)

# Load and find records
data = load_csv_file("path/to/file.csv")
record = find_record_by_field(data, "id", "123")
```

## Integration Points

- **retrieve_agent**: Use ComplianceDocumentProcessor for semantic search
- **transactional_agent**: Use DataLoader for patient data lookups
- **rag_search_tool**: Use utilities for document processing
- **Other agents**: Can reuse any utility functions as needed
