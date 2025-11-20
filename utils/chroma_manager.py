# import os
# import re
# import chromadb
# from chromadb.config import Settings
# from chromadb.utils import embedding_functions
# from PyPDF2 import PdfReader  # or any PDF library you prefer
# from typing import List
# import os
# from dotenv import load_dotenv

# load_dotenv()
# api_key = os.getenv("CHROMA_HUGGINGFACE_API_KEY")

# os.environ["CHROMA_HUGGINGFACE_API_KEY"] = api_key

# print(api_key)


import os
import re
from typing import List
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from PyPDF2 import PdfReader

# -----------------------------
# 1️⃣ Sanitize text
# -----------------------------
def sanitize_text(text: str) -> str:
    """Remove invalid characters and placeholders."""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    # Remove non-printable characters
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E]+", " ", text)
    # Skip invalid or placeholder values
    if text.lower() in ["error", "none", "nan", "null", ""]:
        return ""
    return text

# -----------------------------
# 2️⃣ Extract text from a PDF
# -----------------------------
def extract_pdf_text(pdf_path: str) -> List[str]:
    """Extract sanitized text from a PDF."""
    text_blocks = []
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            page_text = page.extract_text()
            clean_text = sanitize_text(page_text)
            if clean_text:
                text_blocks.append(clean_text)
    except Exception as e:
        print(f"⚠️ Failed to read {pdf_path}: {e}")
    return text_blocks

# -----------------------------
# 3️⃣ Extract text from folder
# -----------------------------
def extract_text_from_folder(folder_path: str) -> List[str]:
    """Extract text from all PDFs in a folder."""
    all_texts = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            all_texts.extend(extract_pdf_text(pdf_path))
    return all_texts

# -----------------------------
# 4️⃣ Setup Chroma collection
# -----------------------------
def setup_chroma(text_data: List[str] = None, persist_dir: str = "./chroma_persist"):
    # Initialize Chroma client
    client = chromadb.Client(
        Settings(
            persist_directory=persist_dir,
            anonymized_telemetry=False
        )
    )

    # Use a local sentence-transformers model (no API key required)
    ef = embedding_functions.HuggingFaceEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    collection = client.get_or_create_collection(
        name="logs_collection",
        embedding_function=ef
    )

    # Add documents safely
    if text_data:
        clean_texts, ids = [], []
        for i, text in enumerate(text_data):
            text = sanitize_text(text)
            if not text:
                continue

            # Chunk large text
            max_length = 1000
            if len(text) > max_length:
                parts = [text[j:j+max_length] for j in range(0, len(text), max_length)]
                for k, part in enumerate(parts):
                    clean_texts.append(part)
                    ids.append(f"doc{i}_part{k}")
            else:
                clean_texts.append(text)
                ids.append(f"doc{i}")

        # Attempt to add documents, skip failures
        if clean_texts:
            for idx, doc in zip(ids, clean_texts):
                try:
                    collection.add(ids=[idx], documents=[doc])
                except Exception as e:
                    print(f"⚠️ Skipped embedding for document {idx}: {e}")
        else:
            print("⚠️ No valid text data to add.")

    return client, collection

# -----------------------------
# 5️⃣ Query Chroma safely
# -----------------------------
def query_chroma(query: str, client, collection_name="logs_collection", n_results=3):
    if not hasattr(client, "get_collection"):
        raise TypeError("client must be a valid chromadb.Client instance")

    clean_query = sanitize_text(query)
    if not clean_query:
        raise ValueError("Query text is invalid after sanitization")

    collection = client.get_collection(name=collection_name)

    # Attempt query, skip embedding failures
    try:
        results = collection.query(query_texts=[clean_query], n_results=n_results)
    except Exception as e:
        print(f"⚠️ Query embedding failed: {e}")
        results = {}
    return results

# -----------------------------
# 6️⃣ Example usage
# -----------------------------
# if __name__ == "__main__":
#     folder_path = "pdf_folder"  # Replace with your folder

#     # Extract all texts
#     all_texts = extract_text_from_folder(folder_path)
#     print(f"Extracted {len(all_texts)} text blocks.")

#     # Setup Chroma collection
#     chroma_client, collection = setup_chroma(all_texts)

#     # Query example
#     query = "network logs"
#     results = query_chroma(query, chroma_client)
#     print("Query results:", results)
