from pypdf import PdfReader
from docx import Document


def extract_text(file_path: str, filename: str) -> str:
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    if ext == "pdf":
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()

    if ext == "docx":
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs).strip()

    raise ValueError(f"Unsupported resume file type: .{ext}. Please upload a PDF or DOCX file.")
