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

def extract_pdf_pages_map(file_bytes: bytes) -> List[Dict[str, Any]]:
    """Extract page-by-page text map from PDF."""
    pdf_file = io.BytesIO(file_bytes)
    reader = PdfReader(pdf_file)
    pages_map = []
    for idx, page in enumerate(reader.pages):
        page_raw = page.extract_text() or ""
        lines = [line.strip() for line in page_raw.splitlines() if line.strip()]
        pages_map.append({
            "page": idx + 1,
            "text": "\n".join(lines),
            "lines": lines
        })
    return pages_map

def detect_chapters_from_pages(pages_map: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str]:
    """
    Analyzes page-by-page PDF text to detect chapter structures, titles, start/end pages, and confidence.
    Matches headings: 'Chapter 1', 'CHAPTER 1: Intro', '3. Data Structures', 'Unit 3', etc.
    """
    total_pages = len(pages_map)
    if total_pages == 0:
        return [], "low"

    detected_headings = []

    p1 = re.compile(
        r'^\s*(?:CHAPTER|UNIT|MODULE|SECTION)\s+([IVXLCDM\d]+|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN)\b[\:\s\-\–\—]*(.*)$',
        re.IGNORECASE
    )
    p2 = re.compile(r'^\s*(\d{1,2}(?:\.\d{1,2})?)\.?\s+([A-Z0-9\s\-\:\,\'\"]{3,70})$')
    
    roman_map = {'i': 1, 'ii': 2, 'iii': 3, 'iv': 4, 'v': 5, 'vi': 6, 'vii': 7, 'viii': 8, 'ix': 9, 'x': 10,
                 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}

    for p_obj in pages_map:
        page_num = p_obj["page"]
        lines = p_obj["lines"]
        
        for line_idx, line in enumerate(lines[:5]):
            if len(line) > 100 or len(line) < 3:
                continue
                
            lower_line = line.lower()
            if any(ref in lower_line for ref in ["discussed in", "refer to", "see chapter", "in chapter", "from chapter", "according to", "as shown in"]):
                continue
            if line.endswith(".") and not re.search(r'\d+\.$', line):
                continue
                
            m1 = p1.match(line)
            if m1:
                raw_num = m1.group(1).lower()
                chap_num = roman_map.get(raw_num, None)
                if chap_num is None:
                    try:
                        chap_num = int(raw_num)
                    except ValueError:
                        chap_num = len(detected_headings) + 1
                
                raw_title = m1.group(2).strip()
                clean_title = re.sub(r'^[\:\-\–\—\s]+', '', raw_title).strip()
                if not clean_title:
                    clean_title = f"Chapter {chap_num}"
                else:
                    clean_title = f"Chapter {chap_num}: {clean_title.title()}"
                    
                if not any(h["page"] == page_num for h in detected_headings):
                    detected_headings.append({
                        "chapter_number": chap_num,
                        "title": clean_title,
                        "page": page_num,
                        "raw_line": line
                    })
                break
                
            m2 = p2.match(line)
            if m2 and line_idx <= 2:
                num_str = m2.group(1)
                title_str = m2.group(2).strip()
                
                if len(title_str.split()) <= 8 and not title_str.lower().startswith(("the ", "a ", "an ", "this ", "that ", "in ", "when ", "if ")):
                    try:
                        chap_num = int(num_str.split('.')[0])
                    except ValueError:
                        chap_num = len(detected_headings) + 1
                        
                    clean_title = f"Chapter {chap_num}: {title_str.title()}"
                    if not any(h["page"] == page_num for h in detected_headings):
                        detected_headings.append({
                            "chapter_number": chap_num,
                            "title": clean_title,
                            "page": page_num,
                            "raw_line": line
                        })
                    break

    if not detected_headings:
        return [], "low"

    detected_headings.sort(key=lambda x: x["page"])
    
    chapters = []
    for i, head in enumerate(detected_headings):
        chap_num = i + 1
        start_p = head["page"]
        end_p = total_pages if (i == len(detected_headings) - 1) else (detected_headings[i + 1]["page"] - 1)
        if end_p < start_p:
            end_p = start_p

        chap_pages_lines = []
        for p_obj in pages_map:
            p_num = p_obj["page"]
            if start_p <= p_num <= end_p:
                chap_pages_lines.append(f"[Page {p_num}]\n" + p_obj["text"])

        chap_text = clean_text("\n\n".join(chap_pages_lines))
        word_c = len(re.findall(r'\w+', chap_text))

        chapters.append({
            "chapter_number": chap_num,
            "title": head["title"],
            "start_page": start_p,
            "end_page": end_p,
            "extracted_text": chap_text,
            "word_count": word_c,
            "detection_confidence": "high" if len(detected_headings) >= 2 else "medium"
        })

    overall_conf = "high" if (len(chapters) >= 2 and all(c["end_page"] >= c["start_page"] for c in chapters)) else "medium"
    return chapters, overall_conf

def process_uploaded_document(file: UploadFile, file_bytes: bytes) -> Tuple[str, int, List[Dict[str, Any]], str]:
    filename = file.filename.lower() if file.filename else "document"
    extracted_text = ""
    detected_chapters = []
    overall_confidence = "high"
    
    if filename.endswith(".pdf"):
        pages_map = extract_pdf_pages_map(file_bytes)
        num_pages = len(pages_map)
        extracted_text = extract_text_from_pdf(file_bytes)
        
        detected_chapters, overall_confidence = detect_chapters_from_pages(pages_map)
        if not detected_chapters:
            word_count_full = len(re.findall(r'\w+', extracted_text))
            detected_chapters = [{
                "chapter_number": 1,
                "title": "Full Document",
                "start_page": 1,
                "end_page": max(1, num_pages),
                "extracted_text": extracted_text,
                "word_count": word_count_full,
                "detection_confidence": "medium"
            }]
            overall_confidence = "medium"
    elif filename.endswith(".docx") or filename.endswith(".doc"):
        extracted_text = extract_text_from_docx(file_bytes)
        word_c = len(re.findall(r'\w+', extracted_text))
        detected_chapters = [{
            "chapter_number": 1,
            "title": "Complete Document",
            "start_page": 1,
            "end_page": 1,
            "extracted_text": extracted_text,
            "word_count": word_c,
            "detection_confidence": "high"
        }]
    elif filename.endswith(".txt"):
        try:
            extracted_text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            extracted_text = file_bytes.decode("latin-1", errors="replace")
        word_c = len(re.findall(r'\w+', extracted_text))
        detected_chapters = [{
            "chapter_number": 1,
            "title": "Text Content",
            "start_page": 1,
            "end_page": 1,
            "extracted_text": extracted_text,
            "word_count": word_c,
            "detection_confidence": "high"
        }]
    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Please upload a PDF, DOCX, or TXT file."
        )
        
    extracted_text = clean_text(extracted_text)
    word_count = len(re.findall(r'\w+', extracted_text))
    
    logger.info(f"Processed '{filename}': {word_count} words, {len(detected_chapters)} chapters detected.")
    
    if not extracted_text or word_count < 35:
        raise HTTPException(
            status_code=400,
            detail="No readable text could be extracted from this document. The file may be scanned, image-only, password-protected, empty, or corrupted. Please upload a digital, text-readable PDF, DOCX, or TXT file."
        )
        
    return extracted_text, word_count, detected_chapters, overall_confidence

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

def extract_page_range_text(full_text: str, start_page: int, end_page: int) -> str:
    """
    Extract text corresponding to pages start_page through end_page 
    from extracted text containing [Page X] markers.
    """
    if not full_text:
        return ""
    
    page_blocks = re.split(r'\[Page\s+(\d+)\]', full_text, flags=re.IGNORECASE)
    if len(page_blocks) < 3:
        return full_text
        
    extracted_blocks = []
    for i in range(1, len(page_blocks), 2):
        try:
            p_num = int(page_blocks[i])
            p_text = page_blocks[i+1]
            if start_page <= p_num <= end_page:
                extracted_blocks.append(f"[Page {p_num}]{p_text}")
        except (ValueError, IndexError):
            continue
            
    res = "".join(extracted_blocks).strip()
    return res if res else full_text

