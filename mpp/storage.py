"""Attachment file handling — save uploaded files and classify them for preview."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from . import config

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_name(name: str) -> str:
    name = Path(name).name  # strip any path components
    cleaned = _SAFE.sub("_", name).strip("_")
    return cleaned or "file"


def save_bytes(experiment_id: int, filename: str, data: bytes, content_type: str = "") -> dict:
    """Persist raw bytes as an attachment for an experiment. Returns metadata dict."""
    dest_dir = config.ATTACH_DIR / str(experiment_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    safe = _safe_name(filename)
    path = dest_dir / safe
    # Avoid clobbering an existing file with the same name.
    if path.exists():
        stem, suffix = path.stem, path.suffix
        i = 1
        while path.exists():
            path = dest_dir / f"{stem}_{i}{suffix}"
            i += 1
    path.write_bytes(data)
    return {
        "filename": safe,
        "path": str(path),
        "content_type": content_type,
        "size": len(data),
    }


def save_upload(experiment_id: int, uploaded_file) -> dict:
    """Persist a Streamlit UploadedFile-like object (.name, .type, .getbuffer())."""
    data = uploaded_file.getbuffer()
    return save_bytes(
        experiment_id,
        uploaded_file.name,
        bytes(data),
        getattr(uploaded_file, "type", "") or "",
    )


def kind(filename: str) -> str:
    """Bucket a filename for preview: image | pdf | excel | text | other."""
    ext = Path(filename).suffix.lower()
    if ext in config.IMAGE_EXTS:
        return "image"
    if ext in config.PDF_EXTS:
        return "pdf"
    if ext in config.EXCEL_EXTS:
        return "excel"
    if ext in config.TEXT_EXTS:
        return "text"
    return "other"


def pdf_text_preview(path: str, max_chars: int = 2000) -> Optional[str]:
    """Best-effort text extraction for previewing a PDF; returns None on failure."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(path)
        out = []
        for page in reader.pages[:5]:
            out.append(page.extract_text() or "")
        text = "\n".join(out).strip()
        return text[:max_chars] if text else None
    except Exception:
        return None
