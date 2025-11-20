import os
import re
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()
#from sentence_transformers import SentenceTransformer
#from langchain_core.documents import Document
#from langchain.schema.Document import Document
#from langchain_core.documents import Document
#from langchain_classic.chains import LLMChain
#from langchain_classic.chains import RetrievalQA
#from langchain_core.prompts import PromptTemplate
#from langchain.vectorstores.chroma import Chroma  # Chroma for vector storage

tavily= TavilyClient()

# -----------------------------
# 1️⃣ Sanitize text
# -----------------------------
def sanitize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    # Remove non-printable characters
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E]+", " ", text)
    if text.lower() in ["error", "none", "nan", "null", ""]:
        return ""
    return text

# -----------------------------
# 2️⃣ Build retriever from text
# -----------------------------
'''
def build_retriever(all_texts: List[str], persist_dir: str = "./chroma_persist"):
    # Convert to Document objects
    docs = [Document(page_content=sanitize_text(t)) for t in all_texts if sanitize_text(t)]

    # Split large text into chunks
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    splits = splitter.split_documents(docs)

    # Hugging Face model (SentenceTransformer) for embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Get embeddings for all document chunks
    embeddings = [model.encode(doc.page_content) for doc in splits]

    # Build Chroma vector store (check Chroma installation)
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    return vectordb.as_retriever(search_kwargs={"k": 3})

# -----------------------------
# 3️⃣ Get Google Gemini LLM
# -----------------------------
def get_llm():
    # Google Gemini integration with langchain_google_genai
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4)

# -----------------------------
# 4️⃣ Build QA chain
# -----------------------------
def build_qa_chain(retriever):
    llm = get_llm()

    prompt_template = """
    You are a helpful assistant. Use the context below to answer the question.
    If you don’t know the answer, say "I’m sorry, the answer isn’t available in the uploaded documents."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Build RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

'''
# -----------------------------
# 5️⃣ Example usage
# -----------------------------
# if __name__ == "__main__":
#     # Example list of texts (replace with PDF extraction)
#     all_texts = [
#         "This is the first document about network logs.",
#         "Second document explaining troubleshooting steps.",
#         "Another document with KPI metrics for 5G network."
#     ]

#     # Build retriever
#     retriever = build_retriever(all_texts)

#     # Build QA chain
#     qa_chain = build_qa_chain(retriever)

#     # Ask a question
#     query = "Explain the troubleshooting steps for network issues"
#     result = qa_chain.run(query)
#     print("Answer:", result)

def search(query: str) -> str:
    """
    Searches for an answer to the given query using the provided texts.

    Args:
        query (str): The query to search for.

    Returns:
        str: The answer to the query, or an error message if no answer is found.
    """
#    return f"Found the result for your network query '{query}'"
    return tavily.search(query=query)
