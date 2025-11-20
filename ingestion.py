import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import Docx2txtLoader
import docx2txt

load_dotenv()

# Configure Google API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")
os.environ["GOOGLE_API_KEY"] = api_key


def load_files(uploaded_files):
    for file in uploaded_files:
        # Save uploaded file to disk
        file_path = f"temp_{file.name}"
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        
        ext = file.name.split(".")[-1].lower()
        if ext == "pdf":
            pdf_loader(file_path)
        elif ext == "docx":
            doc_loader(file_path)
        elif ext == "csv":
            csv_loader(file_path)
        elif ext == "txt":
            text_loader(file_path)
        else:
            return "Invalid file format"
        
        # Clean up temp file
        os.remove(file_path)
    return "Files loaded successfully"

def pdf_loader(file):
    document = PyPDFLoader(file)
    documents= document.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator="\n")
    docs = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index_pdf")
    new_vectorstore = FAISS.load_local("faiss_index_pdf", embeddings, allow_dangerous_deserialization=True)

def doc_loader(file):
    document = Docx2txtLoader(file)
    documents= document.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator="\n")
    docs = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index_docx")
    new_vectorstore = FAISS.load_local("faiss_index_docx", embeddings, allow_dangerous_deserialization=True)

def csv_loader(file):
    document = CSVLoader(file)
    documents= document.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator="\n")
    docs = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index_csv")

def text_loader(file):
    document = TextLoader(file)
    documents = document.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator="\n")
    docs = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index_txt")

if __name__ == '__main__':
    print("Data Ingestion...")