import io
import re
import logging
from typing import Tuple
from pypdf import PdfReader
import docx
from fastapi import UploadFile, HTTPException

logger = logging.getLogger("doc_processor")

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract readable text from PDF with header/footer deduplication and multi-page preservation."""
    try:
        pdf_file = io.BytesIO(file_bytes)
        reader = PdfReader(pdf_file)
        num_pages = len(reader.pages)
        
        page_lines_list = []
        for idx, page in enumerate(reader.pages):
            page_raw = page.extract_text() or ""
            lines = [line.strip() for line in page_raw.splitlines() if line.strip()]
            page_lines_list.append(lines)

        # Detect repeated lines across pages (headers & footers) if multi-page
        # Normalize lines by removing trailing digits/page numbers so "Title 2" and "Title 3" match
        line_occurrences = {}
        if num_pages > 1:
            for lines in page_lines_list:
                normalized_lines = set()
                for l in lines:
                    norm = re.sub(r'[\d\s•|\-\.]+$', '', l, flags=re.IGNORECASE).strip()
                    norm = re.sub(r'^(?:page\s*\d+|chapter\s*\d+|part\s*\d+)', '', norm, flags=re.IGNORECASE).strip()
                    if len(norm) > 3:
                        normalized_lines.add(norm.lower())
                for norm in normalized_lines:
                    line_occurrences[norm] = line_occurrences.get(norm, 0) + 1
                    
        cleaned_pages = []
        for p_idx, lines in enumerate(page_lines_list):
            cleaned_page_lines = []
            for line in lines:
                norm = re.sub(r'[\d\s•|\-\.]+$', '', line, flags=re.IGNORECASE).strip()
                norm = re.sub(r'^(?:page\s*\d+|chapter\s*\d+|part\s*\d+)', '', norm, flags=re.IGNORECASE).strip().lower()
                
                # Filter out lines appearing on >30% of pages in multi-page PDF
                if num_pages > 1 and norm and line_occurrences.get(norm, 0) > max(1, num_pages * 0.3):
                    continue
                # Filter out standalone page numbers & common metadata lines
                if re.match(r'^(page\s*\d+(\s*of\s*\d+)?|\d+)$', line, re.IGNORECASE):
                    continue
                # Filter out running titles like "The Garden Beyond the Stars 2." or "The Clockmaker of Riverton 4"
                if re.match(r'^[A-Z][A-Za-z0-9\s\-_:\'",.]{3,60}\s+\d{1,3}\.?$', line):
                    continue
                    
                cleaned_page_lines.append(line)
            
            if cleaned_page_lines:
                cleaned_pages.append(f"[Page {p_idx + 1}]\n" + "\n".join(cleaned_page_lines))

        full_text = "\n\n".join(cleaned_pages)
        full_text = clean_text(full_text)
        if not full_text or len(full_text.strip()) < 10:
            raise HTTPException(
                status_code=400,
                detail="The uploaded PDF contains no extractable text layer. If this is a scanned document, please use OCR or upload a text-readable PDF."
            )
        logger.info(f"Extracted PDF text: {num_pages} pages, {len(full_text)} chars")
        return full_text
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDF Extraction failure: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to extract text from PDF document: {str(e)}"
        )

def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from DOCX Word document."""
    try:
        docx_file = io.BytesIO(file_bytes)
        doc = docx.Document(docx_file)
        full_text = []
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text.strip())
        
        # Also extract table text if present
        for table in doc.tables:
            for row in table.rows:
                row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_text:
                    full_text.append(" | ".join(row_text))
                    
        return "\n".join(full_text)
    except Exception as e:
        logger.error(f"DOCX Extraction failure: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to extract text from Word DOCX file: {str(e)}"
        )

def process_uploaded_document(file: UploadFile, file_bytes: bytes) -> Tuple[str, int]:
    filename = file.filename.lower() if file.filename else "document"
    extracted_text = ""
    
    if filename.endswith(".pdf"):
        extracted_text = extract_text_from_pdf(file_bytes)
    elif filename.endswith(".docx") or filename.endswith(".doc"):
        extracted_text = extract_text_from_docx(file_bytes)
    elif filename.endswith(".txt"):
        try:
            extracted_text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            extracted_text = file_bytes.decode("latin-1", errors="replace")
    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Please upload a PDF, DOCX, or TXT file."
        )
        
    extracted_text = clean_text(extracted_text)
    word_count = len(re.findall(r'\w+', extracted_text))
    
    logger.info(f"Processed '{filename}': {word_count} words extracted. Preview: {extracted_text[:120]!r}")
    
    if not extracted_text or word_count < 35:
        raise HTTPException(
            status_code=400,
            detail="No readable text could be extracted from this document. The file may be scanned, image-only, password-protected, empty, or corrupted. Please upload a digital, text-readable PDF, DOCX, or TXT file."
        )
        
    return extracted_text, word_count

def clean_text(text: str) -> str:
    """Clean unneeded white spaces, control characters, PDF page headers, and testing footers."""
    if not text:
        return ""
    text = re.sub(r'[\r\t]', ' ', text)
    # Ensure newlines between paragraphs have period boundaries if missing
    text = re.sub(r'(?<=[^\n.!?])\n+(?=[A-Z0-9])', '. ', text)
    # Remove lines ending with page numbers like "Story Title 1", "The Clockmaker of Riverton 2."
    text = re.sub(r'^[^\n]*?\b\d+\s*\.?\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^(?!\[Page\s*\d+\])[^\n]+?\s*•?\s*Page\s*\d+.*$', '', text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'(?<!\[)Page\s+\d+(\s+of\s+\d+)?(?!\])', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Prepared as a sample document.*?\.', '', text, flags=re.IGNORECASE)
    text = re.sub(r'for testing PDF.*?\.', '', text, flags=re.IGNORECASE)
    text = re.sub(r'A ten-part short story.*?\.', '', text, flags=re.IGNORECASE)
    text = re.sub(r'A ten-chapter mystery story.*?\.', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()

def chunk_text_if_needed(text: str, max_words: int = 4000) -> str:
    """
    If a chapter document is too long for prompt limits, 
    sample key sections (beginning, middle, end) intelligently 
    so key concepts across the entire chapter are retained.
    """
    words = text.split()
    if len(words) <= max_words:
        return text
        
    # Divide into 3 chunks and take equal parts from start, middle, and end
    chunk_size = max_words // 3
    start_part = " ".join(words[:chunk_size])
    mid_idx = len(words) // 2
    mid_part = " ".join(words[mid_idx - (chunk_size // 2): mid_idx + (chunk_size // 2)])
    end_part = " ".join(words[-chunk_size:])
    
    combined = f"{start_part}\n\n[... Chapter Middle Section ...]\n\n{mid_part}\n\n[... Chapter Final Section ...]\n\n{end_part}"
    return combined
