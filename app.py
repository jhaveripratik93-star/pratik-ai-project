import streamlit as st
import os
from dotenv import load_dotenv
from utils.file_loader import extract_text_from_files
from utils.chroma_manager import setup_chroma, query_chroma
from utils.safety import is_safe_response
import google.generativeai as genai

load_dotenv()

st.set_page_config(page_title="Human-in-the-Loop Chatbot", layout="wide")

api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# --- Sidebar ---
st.sidebar.header("Upload Knowledge Files")
uploaded_files = st.sidebar.file_uploader(
    "Upload multiple documents", type=["pdf", "docx", "csv", "txt"], accept_multiple_files=True
)

# --- Main UI ---
st.title("🤖 AI Chatbot with Human-in-the-Loop & Guardrails")

if uploaded_files:
    with st.spinner("Processing uploaded files..."):
        text_data = extract_text_from_files(uploaded_files)
        chroma_client, collection = setup_chroma(text_data)
    st.success("Files processed and indexed successfully!")

# --- Chat Interface ---
query = st.text_input("Ask a question about your uploaded files:")
if st.button("Generate Answer"):
    if not uploaded_files:
        st.warning("Please upload at least one file first.")
    elif not query.strip():
        st.warning("Enter a question.")
    else:
        # Retrieve context
        docs = query_chroma(collection, query)
        context = "\n\n".join(docs)

        prompt = f"Answer the question using the provided context. If not found, say 'Not found in documents.'\n\nContext:\n{context}\n\nQuestion: {query}"

        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)

        answer = response.text

        # Apply guardrails
        if not is_safe_response(answer):
            st.error("⚠️ The response contained unsafe or disallowed content.")
        else:
            # Human-in-the-loop confirmation
            st.write("### ✍️ AI Suggested Answer")
            st.write(answer)

            st.write("### ✅ Human Review")
            human_feedback = st.text_area("Edit or approve the answer before sending:", value=answer)
            if st.button("Submit Final Answer"):
                st.success("✅ Final answer approved by human:")
                st.write(human_feedback)
