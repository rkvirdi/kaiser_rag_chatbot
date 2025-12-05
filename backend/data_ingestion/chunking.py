"""
Text chunking strategies for document processing in Kaiser RAG system.
Handles various approaches to split documents for vector DB ingestion.
Includes recursive text splitter for semantic coherence and table preservation.
"""

import logging
import re
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def chunk_text_recursive(text: str, chunk_size: int = 500, overlap: int = 100, 
                         separators: List[str] = None) -> List[str]:
    """
    Recursive text chunking that respects semantic boundaries.
    Tries to split on sentence, paragraph, line, then word boundaries.
    Preserves tables and structured content.
    
    Args:
        text: The text to chunk.
        chunk_size: Target number of words per chunk (default: 500).
        overlap: Number of words to overlap between chunks (default: 100).
        separators: List of separators to try in order. Default: ['\n\n', '\n', '. ', ' ']
    
    Returns:
        List of text chunks.
    
    Example:
        >>> text = "Paragraph 1. Sentence 2. More text...\\n\\nParagraph 2..."
        >>> chunks = chunk_text_recursive(text, chunk_size=100, overlap=20)
        >>> len(chunks) > 1
        True
    """
    if not text or not text.strip():
        logger.warning("Empty text provided for chunking")
        return []
    
    if separators is None:
        # Try to split on these in order - respects semantic boundaries
        separators = ['\n\n', '\n', '. ', ' ']
    
    chunks = []
    current_chunk = ""
    words_in_current = 0
    
    def split_recursively(content: str, sep_idx: int = 0) -> List[str]:
        """Recursively split content using separators."""
        if sep_idx >= len(separators):
            # If no more separators, split by words
            words = content.split()
            if len(words) > chunk_size:
                result = []
                for i in range(0, len(words), chunk_size - overlap):
                    result.append(" ".join(words[i:i + chunk_size]))
                return result
            return [content] if content.strip() else []
        
        separator = separators[sep_idx]
        parts = content.split(separator)
        
        # If split was effective (more than 1 part), use this separator
        if len(parts) > 1:
            return parts
        
        # Otherwise, try next separator
        return split_recursively(content, sep_idx + 1)
    
    parts = split_recursively(text)
    
    for part in parts:
        if not part.strip():
            continue
        
        part_words = part.split()
        part_word_count = len(part_words)
        
        # If adding this part exceeds chunk_size, save current chunk and start new one
        if words_in_current + part_word_count > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            
            # Add overlap from end of previous chunk to start of next
            overlap_text = " ".join(part_words[:overlap])
            current_chunk = overlap_text
            words_in_current = overlap
        
        if current_chunk:
            current_chunk += " " + part
        else:
            current_chunk = part
        
        words_in_current = len(current_chunk.split())
    
    # Add final chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    logger.info(f"Created {len(chunks)} chunks using recursive splitting ({len(text.split())} words)")
    return chunks


def extract_tables(text: str) -> tuple[List[Dict[str, Any]], str]:
    """
    Extract tables from text to preserve them during chunking.
    Detects common table patterns (CSV-like, pipe-delimited, etc).
    
    Args:
        text: The text to scan for tables.
    
    Returns:
        Tuple of (extracted_tables, text_without_tables)
        Each table is stored with its metadata for later reinsertion.
    """
    tables = []
    lines = text.split('\n')
    table_lines = []
    text_lines = []
    in_table = False
    
    for line in lines:
        # Detect table markers: lines with pipes (|), dashes, or tab separation
        is_table_line = (
            '|' in line or  # Markdown-style tables
            ('\t' in line and line.count('\t') > 1) or  # Tab-separated
            (re.match(r'^[\s\-|]+$', line))  # Separator lines
        )
        
        if is_table_line:
            if not in_table and table_lines:
                # Save previous table
                tables.append({
                    "content": '\n'.join(table_lines),
                    "type": "table"
                })
                table_lines = []
            
            table_lines.append(line)
            in_table = True
        else:
            if in_table and table_lines:
                # End of table
                tables.append({
                    "content": '\n'.join(table_lines),
                    "type": "table"
                })
                table_lines = []
                in_table = False
            
            text_lines.append(line)
    
    # Save last table if exists
    if table_lines:
        tables.append({
            "content": '\n'.join(table_lines),
            "type": "table"
        })
    
    cleaned_text = '\n'.join(text_lines)
    logger.info(f"Extracted {len(tables)} tables from text")
    return tables, cleaned_text


def chunk_text_with_table_preservation(text: str, chunk_size: int = 500, overlap: int = 100) -> List[Dict[str, Any]]:
    """
    Chunk text while preserving table structure and context.
    Tables are kept intact and marked with metadata.
    
    Args:
        text: The text to chunk.
        chunk_size: Target number of words per chunk.
        overlap: Number of words to overlap between chunks.
    
    Returns:
        List of chunk dictionaries with type and metadata.
    """
    # Extract tables first
    tables, text_without_tables = extract_tables(text)
    
    # Chunk the text
    chunks = chunk_text_recursive(text_without_tables, chunk_size, overlap)
    
    # Convert to dictionary format and mark tables
    result = []
    for idx, chunk in enumerate(chunks):
        result.append({
            "id": f"chunk_{idx}",
            "content": chunk,
            "type": "text",
            "chunk_index": idx
        })
    
    # Add tables as separate chunks (they should not be split)
    for table_idx, table in enumerate(tables):
        result.append({
            "id": f"table_{table_idx}",
            "content": table["content"],
            "type": "table",
            "chunk_index": table_idx
        })
    
    logger.info(f"Created {len(result)} chunks ({len(chunks)} text + {len(tables)} tables)")
    return result


def chunk_text_by_sentences(text: str, sentences_per_chunk: int = 5) -> List[str]:
    """
    Split text into chunks based on sentence count.
    
    Args:
        text: The text to chunk.
        sentences_per_chunk: Number of sentences per chunk (default: 5).
    
    Returns:
        List of text chunks.
    """
    if not text or not text.strip():
        logger.warning("Empty text provided for sentence chunking")
        return []
    
    # Simple sentence splitting (basic - doesn't handle all cases like "Dr." or "Mr.")
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    chunks = []
    
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = '. '.join(sentences[i:i + sentences_per_chunk]) + '.'
        if chunk.strip() and chunk != '.':
            chunks.append(chunk)
    
    logger.info(f"Created {len(chunks)} sentence-based chunks")
    return chunks


def chunk_text_by_paragraphs(text: str, max_chunk_size: int = 1000) -> List[str]:
    """
    Split text into chunks based on paragraph boundaries.
    
    Args:
        text: The text to chunk.
        max_chunk_size: Maximum characters per chunk (default: 1000).
    
    Returns:
        List of text chunks.
    """
    if not text or not text.strip():
        logger.warning("Empty text provided for paragraph chunking")
        return []
    
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) <= max_chunk_size:
            current_chunk += paragraph + "\n\n"
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = paragraph + "\n\n"
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    logger.info(f"Created {len(chunks)} paragraph-based chunks")
    return chunks


def create_chunks_with_metadata(text: str, source: str, doc_type: str, 
                                chunk_size: int = 500, overlap: int = 100,
                                preserve_tables: bool = True) -> List[Dict[str, Any]]:
    """
    Create text chunks with rich metadata for vector DB ingestion.
    Uses recursive text splitter for semantic coherence.
    Optionally preserves table structure.
    
    Args:
        text: The text to chunk.
        source: Source file or document identifier.
        doc_type: Type of document ('compliance', 'policy', 'guideline').
        chunk_size: Number of words per chunk.
        overlap: Number of words to overlap between chunks.
        preserve_tables: Whether to preserve table structure (default: True).
    
    Returns:
        List of chunk dictionaries with metadata.
    
    Example:
        >>> chunks = create_chunks_with_metadata(
        ...     text="sample text...",
        ...     source="policies/benefit-summary.pdf",
        ...     doc_type="policy"
        ... )
        >>> chunks[0].keys()
        dict_keys(['id', 'content', 'source', 'type', 'chunk_index', ...])
    """
    from datetime import datetime
    
    # Use table-preserving chunking if requested
    if preserve_tables:
        text_chunks = chunk_text_with_table_preservation(text, chunk_size, overlap)
    else:
        # Use recursive splitter for better semantic coherence
        raw_chunks = chunk_text_recursive(text, chunk_size, overlap)
        text_chunks = [{"content": chunk, "type": "text"} for chunk in raw_chunks]
    
    chunks_with_meta = []
    
    for idx, chunk_data in enumerate(text_chunks):
        if isinstance(chunk_data, dict):
            chunk_content = chunk_data.get("content", "")
            chunk_type = chunk_data.get("type", "text")
        else:
            chunk_content = chunk_data
            chunk_type = "text"
        
        chunks_with_meta.append({
            "id": f"{doc_type}_{source.replace('/', '_').replace('.', '_')}_{idx}",
            "content": chunk_content,
            "source": source,
            "type": doc_type,
            "content_type": chunk_type,  # "text" or "table"
            "chunk_index": idx,
            "total_chunks": len(text_chunks),
            "created_at": datetime.utcnow().isoformat()
        })
    
    logger.info(f"Created {len(chunks_with_meta)} chunks with metadata for {source} (preserve_tables={preserve_tables})")
    return chunks_with_meta
