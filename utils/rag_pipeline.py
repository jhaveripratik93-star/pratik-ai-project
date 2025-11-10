from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate

def build_retriever(all_texts, persist_dir="./chroma_persist"):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    docs = [Document(page_content=t) for t in all_texts]

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    splits = splitter.split_documents(docs)

    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    return vectordb.as_retriever(search_kwargs={"k": 3})

def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4)

def build_qa_chain(retriever):
    llm = get_llm()

    prompt_template = """
    You are a helpful assistant. Use the context below to answer the question.
    If you don’t know the answer, say “I’m sorry, the answer isn’t available in the uploaded documents.”

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    combine_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    retrieval_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=combine_chain)
    return retrieval_chain
