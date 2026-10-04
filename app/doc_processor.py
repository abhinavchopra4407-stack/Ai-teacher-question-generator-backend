import io
import re
from typing import Tuple
from pypdf import PdfReader
import docx
from fastapi import UploadFile, HTTPException

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF using PyPDF."""
    text_content = []
    try:
        pdf_file = io.BytesIO(file_bytes)
        reader = PdfReader(pdf_file)
        for idx, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                text_content.append(page_text.strip())
        
        full_text = "\n\n".join(text_content)
        return full_text
    except Exception as e:
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
    if not extracted_text or len(extracted_text.strip()) < 20:
        raise HTTPException(
            status_code=400,
            detail="No readable text found in document. The file might be scanned, empty, image-only, or corrupted."
        )
        
    word_count = len(re.findall(r'\w+', extracted_text))
    return extracted_text, word_count

def clean_text(text: str) -> str:
    """Clean unneeded white spaces, control characters, and PDF page numbers."""
    if not text:
        return ""
    text = re.sub(r'[\r\t]', ' ', text)
    text = re.sub(r'Page\s+\d+(\s+of\s+\d+)?', '', text, flags=re.IGNORECASE)
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
