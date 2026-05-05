"""Plain-text extraction for .txt / .docx / .pdf uploads."""
from __future__ import annotations

import io


def extract_text(filename: str, data: bytes) -> str:
    name = (filename or "").lower()
    if name.endswith(".txt") or not name:
        return data.decode("utf-8", errors="replace")
    if name.endswith(".docx"):
        from docx import Document  # lazy: keep tests independent of python-docx
        doc = Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs if p.text)
    if name.endswith(".pdf"):
        text = _extract_pdf_text(data)
        if not text.strip():
            raise ValueError(
                f"no extractable text in {filename!r} "
                "(scanned PDF? OCR not yet supported)"
            )
        return text
    raise ValueError(f"unsupported file extension: {filename!r}")


def _extract_pdf_text(data: bytes) -> str:
    """Three-tier PDF extraction:

      1. pypdf       — fast text-stream walk. Handles 90%+ of PDFs.
      2. pdfplumber  — pdfminer.six under the hood, more aggressive glyph
                       mapping. Catches PDFs where fonts lack ToUnicode tables.
      3. OCR         — pypdfium2 renders pages → PIL images → pytesseract.
                       Last resort for scanned/image-only PDFs (~1-3s/page).

    Tiers are tried in order; we only pay each tier's cost when the previous
    one returned nothing. Native-text PDFs stay millisecond-fast.
    """
    from pypdf import PdfReader  # lazy: keep tests independent of pypdf
    reader = PdfReader(io.BytesIO(data))
    pages = [(p.extract_text() or "") for p in reader.pages]
    text = "\n".join(s for s in (p.strip() for p in pages) if s)
    if text.strip():
        return text

    try:
        import pdfplumber  # lazy: only imported when pypdf came up empty
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            pages = [(p.extract_text() or "") for p in pdf.pages]
        text = "\n".join(s for s in (p.strip() for p in pages) if s)
        if text.strip():
            return text
    except ImportError:
        pass

    return _ocr_pdf(data)


def _ocr_pdf(data: bytes) -> str:
    """Render each PDF page to a PIL image and OCR with Tesseract.

    Renders at 2× scale (≈144 DPI from the default 72 DPI) — empirically the
    sweet spot for typed-text scans: low enough to keep latency reasonable,
    high enough that Tesseract's LSTM resolves serifs and small punctuation.
    Returns "" if any dependency is missing so the caller can raise its own
    "no extractable text" error rather than leak an ImportError to the user.
    """
    try:
        import pypdfium2 as pdfium
        import pytesseract
    except ImportError:
        return ""
    pages = []
    pdf = pdfium.PdfDocument(io.BytesIO(data))
    try:
        for i in range(len(pdf)):
            page = pdf[i]
            pil = page.render(scale=2).to_pil()
            page.close()
            txt = (pytesseract.image_to_string(pil) or "").strip()
            if txt:
                pages.append(txt)
    finally:
        pdf.close()
    return "\n".join(pages)
