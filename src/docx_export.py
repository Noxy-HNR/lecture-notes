"""Mirrors the markdown notes into a matching .docx file, appending each lecture's
section the same way notes.py appends to the .md file, so both stay in sync."""
import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH

NOTES_DIR = Path(__file__).resolve().parent.parent / "notes"

_INLINE_TOKEN = re.compile(r"(\*\*.+?\*\*|\*.+?\*)")


def docx_path(class_code: str) -> Path:
    safe = class_code.replace(" ", "_")
    return NOTES_DIR / f"{safe}.docx"


def _add_inline_runs(paragraph, text: str):
    """Splits on **bold** / *italic* markers and adds runs with the right formatting."""
    for token in _INLINE_TOKEN.split(text):
        if not token:
            continue
        if token.startswith("**") and token.endswith("**"):
            paragraph.add_run(token[2:-2]).bold = True
        elif token.startswith("*") and token.endswith("*"):
            paragraph.add_run(token[1:-1]).italic = True
        else:
            paragraph.add_run(token)


def _append_markdown(doc: Document, markdown: str):
    for raw_line in markdown.split("\n"):
        line = raw_line.rstrip()
        if not line.strip():
            continue

        if line.startswith("### "):
            doc.add_heading(line[4:].strip(), level=3)
        elif line.startswith("## "):
            doc.add_heading(line[3:].strip(), level=2)
        elif line.startswith("# "):
            doc.add_heading(line[2:].strip(), level=1)
        elif re.match(r"^\s*[-*]\s+", line):
            indent = len(line) - len(line.lstrip())
            content = re.sub(r"^\s*[-*]\s+", "", line)
            style = "List Bullet 2" if indent >= 2 else "List Bullet"
            p = doc.add_paragraph(style=style)
            _add_inline_runs(p, content)
        else:
            p = doc.add_paragraph()
            if line.startswith("*(") and line.endswith(")*"):
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            _add_inline_runs(p, line)


def append_section(class_code: str, class_title: str, section_markdown: str):
    """Appends one lecture's formatted section to notes/<CODE>.docx, creating it if needed."""
    path = docx_path(class_code)
    if path.exists():
        doc = Document(str(path))
    else:
        doc = Document()
        doc.add_heading(f"{class_title} ({class_code})", level=1)

    _append_markdown(doc, section_markdown)
    doc.save(str(path))
    return path


def rebuild(class_code: str, class_title: str, full_markdown: str):
    """Rebuilds notes/<CODE>.docx from scratch to match `full_markdown` exactly. Used
    after condensing a session's notes, since incremental append can't handle edits
    or removals to already-written content - only a full rebuild can."""
    doc = Document()
    _append_markdown(doc, full_markdown)
    path = docx_path(class_code)
    doc.save(str(path))
    return path
