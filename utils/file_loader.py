import io
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document

def extract_text_from_files(files):
    text_data = []
    for file in files:
        ext = file.name.split(".")[-1].lower()
        if ext == "pdf":
            reader = PdfReader(file)
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
        elif ext == "docx":
            doc = Document(file)
            text = "\n".join([p.text for p in doc.paragraphs])
        elif ext == "csv":
            df = pd.read_csv(file)
            text = df.to_string()
        elif ext == "txt":
            text = file.read().decode("utf-8")
        else:
            text = ""
        text_data.append(text)
    return text_data
