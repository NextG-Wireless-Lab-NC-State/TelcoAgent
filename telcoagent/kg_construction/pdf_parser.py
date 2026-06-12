"""3GPP PDF Section Parser for TelcoAgent-RAG.

Parses 3GPP specification PDFs (e.g. TS 28.552, TS 28.554, TS 38.211-38.331,
TS 38.901) into structured section chunks using pdfplumber. No LLM required.

Section boundaries are detected via 3GPP clause numbering patterns (e.g. 5.1.1.2.1).
"""

import logging
import re
from pathlib import Path
from typing import List, Optional

import pdfplumber

from .schema import SectionChunk

logger = logging.getLogger(__name__)

# 3GPP section number pattern: starts with digit, followed by dot-separated digits
# Must be at the start of a line and followed by a title (non-digit text)
_SECTION_RE = re.compile(
    r"^(\d+(?:\.\d+)+)\s+(.+?)$",
    re.MULTILINE,
)

# Spec identifier extraction from filename
# Supports two formats:
#   ETSI-style:   "ts_128552v190500p.pdf" -> "TS 28.552"
#   Unified-style: "TS_38.214.pdf"         -> "TS 38.214"
_SPEC_FILENAME_RE_ETSI = re.compile(
    r"ts_1(\d{2})(\d{3})v\d+p",
    re.IGNORECASE,
)
_SPEC_FILENAME_RE_UNIFIED = re.compile(
    r"ts[_\s]?(\d{2})\.(\d{3})",
    re.IGNORECASE,
)


def _extract_spec_id(filename: str) -> str:
    """Extract spec identifier from 3GPP PDF filename.

    Examples:
        ts_128552v190500p.pdf -> TS 28.552
        ts_138214v180700p.pdf -> TS 38.214
        TS_28.552.pdf         -> TS 28.552
        TS_38.211.pdf         -> TS 38.211
    """
    m = _SPEC_FILENAME_RE_ETSI.search(filename)
    if m:
        return f"TS {m.group(1)}.{m.group(2)}"
    m = _SPEC_FILENAME_RE_UNIFIED.search(filename)
    if m:
        return f"TS {m.group(1)}.{m.group(2)}"
    return filename


def _extract_text(pdf_path: str) -> List[dict]:
    """Extract text and tables from each page of the PDF.

    Returns list of dicts with 'page', 'text', and 'tables' keys.
    Pages are processed one at a time and flushed to keep memory low
    for large PDFs (e.g. TS 38.133, 44 MB / 900+ pages).
    """
    import gc

    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            tables = []
            for table in page.extract_tables():
                if table:
                    rows = []
                    for row in table:
                        cells = [str(c).strip() if c else "" for c in row]
                        rows.append(" | ".join(cells))
                    tables.append("\n".join(rows))
            pages.append({"page": i, "text": text, "tables": tables})
            page.flush_cache()
            if i % 100 == 0:
                gc.collect()
                logger.info(f"  Parsed page {i}/{total}")
    return pages


def _merge_pages(pages: List[dict]) -> str:
    """Merge all page texts into a single document string."""
    parts = []
    for p in pages:
        parts.append(p["text"])
    return "\n".join(parts)


def _build_page_index(pages: List[dict]) -> List[int]:
    """Build a character offset -> page number index."""
    index = []
    offset = 0
    for p in pages:
        length = len(p["text"]) + 1  # +1 for join newline
        index.append((offset, offset + length, p["page"]))
        offset += length
    return index


def _offset_to_page(offset: int, page_index: List) -> int:
    """Convert character offset to page number."""
    for start, end, page in page_index:
        if start <= offset < end:
            return page
    return 0


def _split_long_body(
    body: str,
    max_chars: int,
    overlap: int = 200,
) -> List[str]:
    """Split a long section body into sub-chunks at paragraph/sentence boundaries.

    Tries paragraph breaks first, then sentence boundaries, finally a hard cut.
    Adds a small overlap between sub-chunks to preserve context.
    """
    if len(body) <= max_chars:
        return [body]

    parts: List[str] = []
    remaining = body
    while len(remaining) > max_chars:
        window = remaining[:max_chars]
        # Prefer paragraph break, then sentence end, else hard cut.
        cut = window.rfind("\n\n")
        if cut < max_chars // 2:
            cut = max(window.rfind(". "), window.rfind(".\n"))
        if cut < max_chars // 2:
            cut = max_chars
        parts.append(remaining[:cut].strip())
        next_start = max(0, cut - overlap)
        remaining = remaining[next_start:]
    if remaining.strip():
        parts.append(remaining.strip())
    return [p for p in parts if p]


def parse_pdf(
    pdf_path: str,
    min_body_length: int = 50,
    max_section_depth: Optional[int] = None,
    max_section_chars: Optional[int] = 15000,
) -> List[SectionChunk]:
    """Parse a 3GPP PDF into section chunks.

    Args:
        pdf_path: Path to the 3GPP PDF file.
        min_body_length: Minimum section body length to include.
        max_section_depth: Max clause depth (e.g. 4 means up to x.x.x.x).
            None means no limit.
        max_section_chars: Max characters per chunk body. Sections larger
            than this are split into sub-chunks (clause "X.Y/1", "X.Y/2", ...)
            so each chunk fits within the LLM context window. Set to None
            to disable splitting.

    Returns:
        List of SectionChunk objects.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    spec_id = _extract_spec_id(path.name)
    logger.info(f"Parsing {spec_id} from {path.name}")

    # Extract all pages
    pages = _extract_text(str(path))
    full_text = _merge_pages(pages)
    page_index = _build_page_index(pages)

    # Collect all tables by page
    tables_by_page = {}
    for p in pages:
        if p["tables"]:
            tables_by_page[p["page"]] = p["tables"]

    # Find all section headers
    matches = list(_SECTION_RE.finditer(full_text))
    if not matches:
        logger.warning(f"No sections found in {path.name}")
        return []

    logger.info(f"Found {len(matches)} section headers in {path.name}")

    # Build section chunks
    chunks = []
    for i, match in enumerate(matches):
        clause = match.group(1)
        title = match.group(2).strip()

        # Filter by depth if specified
        if max_section_depth is not None:
            depth = clause.count(".") + 1
            if depth > max_section_depth:
                continue

        # Extract body (text between this header and the next)
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        body = full_text[body_start:body_end].strip()

        if len(body) < min_body_length:
            continue

        # Determine page
        page_num = _offset_to_page(match.start(), page_index)

        # Collect tables from this section's pages
        section_tables = []
        end_page = _offset_to_page(body_end - 1, page_index) if body_end > 0 else page_num
        for pg in range(page_num, end_page + 1):
            if pg in tables_by_page:
                section_tables.extend(tables_by_page[pg])

        # Split oversized bodies into sub-chunks so each fits in the LLM
        # context window. Tables stay attached to the first sub-chunk only.
        if max_section_chars and len(body) > max_section_chars:
            sub_bodies = _split_long_body(body, max_section_chars)
            for j, sub_body in enumerate(sub_bodies, start=1):
                chunks.append(
                    SectionChunk(
                        spec=spec_id,
                        clause=f"{clause}/{j}",
                        title=f"{title} (part {j}/{len(sub_bodies)})",
                        body=sub_body,
                        tables=section_tables if j == 1 else [],
                        page_start=page_num,
                    )
                )
        else:
            chunks.append(
                SectionChunk(
                    spec=spec_id,
                    clause=clause,
                    title=title,
                    body=body,
                    tables=section_tables,
                    page_start=page_num,
                )
            )

    logger.info(f"Extracted {len(chunks)} section chunks from {spec_id}")
    return chunks


def parse_multiple_pdfs(
    pdf_paths: List[str],
    min_body_length: int = 50,
    max_section_depth: Optional[int] = None,
    max_section_chars: Optional[int] = 15000,
) -> List[SectionChunk]:
    """Parse multiple 3GPP PDFs into section chunks.

    Args:
        pdf_paths: List of paths to PDF files.
        min_body_length: Minimum section body length.
        max_section_depth: Max clause depth.
        max_section_chars: Max chars per chunk body before splitting.

    Returns:
        Combined list of SectionChunk objects from all PDFs.
    """
    all_chunks = []
    for pdf_path in pdf_paths:
        try:
            chunks = parse_pdf(
                pdf_path,
                min_body_length=min_body_length,
                max_section_depth=max_section_depth,
                max_section_chars=max_section_chars,
            )
            all_chunks.extend(chunks)
        except Exception as e:
            logger.error(f"Failed to parse {pdf_path}: {e}")
    return all_chunks
