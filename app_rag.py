import streamlit as st
from dotenv import load_dotenv
from utils.loader import extract_text
from utils.safety import is_safe
from utils.rag_pipeline import build_retriever, build_qa_chain

load_dotenv()

st.set_page_config(page_title="Doc Chatbot", layout="wide", page_icon="🤖")
st.title("🤖 Document Chatbot with Human-in-the-Loop")

st.sidebar.header("📂 Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs, DOCX, CSV, TXT",
    type=["pdf", "docx", "csv", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("Processing documents..."):
        all_texts = [extract_text(f) for f in uploaded_files]
        retriever = build_retriever(all_texts)
        qa_chain = build_qa_chain(retriever)
    st.sidebar.success("✅ Documents indexed!")
else:
    qa_chain = None

query = st.text_input("Ask a question about your uploaded documents:")

if st.button("Generate Answer"):
    if qa_chain is None:
        st.warning("Please upload documents first.")
    elif not query.strip():
        st.warning("Enter a question.")
    else:
        with st.spinner("Fetching answer..."):
            try:
                res = qa_chain.invoke({"input": query})
                answer = res.get("output", "")
            except Exception as e:
                st.error(f"Error generating response: {e}")
                answer = None

        if answer:
            if not is_safe(answer):
                st.error("⚠️ Unsafe content detected.")
            else:
                st.markdown("### 🤖 AI Proposed Answer")
                st.write(answer)

                st.markdown("---\n### 🧍 Human Review")
                human_edit = st.text_area("Edit or approve:", value=answer)
                if st.button("Submit Final Answer"):
                    st.success("✅ Final answer approved:")
                    st.write(human_edit)
        else:
            st.warning("No answer generated — try rephrasing your question.")
