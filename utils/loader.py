import pandas as pd
from PyPDF2 import PdfReader
from docx import Document

def extract_text(file):
    ext = file.name.split('.')[-1].lower()
    if ext == "pdf":
        reader = PdfReader(file)
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    elif ext == "docx":
        doc = Document(file)
        return "\n".join([p.text for p in doc.paragraphs])
    elif ext == "csv":
        df = pd.read_csv(file)
        return df.to_string()
    elif ext == "txt":
        return file.read().decode("utf-8")
    else:
        return ""
