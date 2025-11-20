import streamlit as st
import time
import os
import base64
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langchain.tools import tool, BaseTool
from callbacks import AgentCallbackHandler
from utils.ingestion import load_files
from utils.guardrails import validate_user_input, sanitize_input, is_network_related
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic import hub


load_dotenv()

@tool
def format_gemini_response(messages):
    """
    Extracts and formats the actual answer from a list of Gemini chat messages.

    Args:
        messages (list): List of messages returned by Gemini API (HumanMessage, AIMessage, ToolMessage, etc.)

    Returns:
        str: Concatenated text of all AI responses.
    """
    formatted_text = []

    for msg in messages:
        # Human messages usually contain the user's input
        if isinstance(msg, HumanMessage):
            formatted_text.append(f"User: {msg.content if isinstance(msg.content, str) else ' '.join(msg.content)}")

        # AIMessage can contain function calls or direct answers
        elif isinstance(msg, AIMessage):
            # If content is not empty
            if msg.content:
                formatted_text.append(f"AI: {msg.content}")
            # If function_call exists
            elif 'function_call' in msg.additional_kwargs:
                fc = msg.additional_kwargs['function_call']
                formatted_text.append(f"AI (Function Call: {fc.get('name')}): {fc.get('arguments')}")

        # ToolMessage usually contains structured responses from tools
        elif isinstance(msg, ToolMessage):
            try:
                import json
                content = msg.content
                # Try parsing JSON if possible
                data = json.loads(content)
                if "output" in data:
                    formatted_text.append(f"AI (Tool Output): {data['output']}")
            except Exception:
                # fallback
                formatted_text.append(f"AI (ToolMessage): {msg.content}")

    # Join all messages with line breaks
    return "\n".join(formatted_text)

def find_tool_by_name(tools: List[BaseTool], tool_name: str) -> BaseTool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool wtih name {tool_name} not found")


st.set_page_config(
    page_title="network issue trouble shooter Style Chat",
    layout="wide"
)

# -----------------------------
# SESSION STATE INIT
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []


# -----------------------------
# SIDEBAR — FILE UPLOADS
# -----------------------------
st.sidebar.title("📂 Uploaded Files")

# Hide default file display with comprehensive CSS
st.sidebar.markdown("""
<style>
.uploadedFile, .stFileUploader > div > div > div > div:nth-child(2) {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

uploaded = st.sidebar.file_uploader(
    "Upload multiple files",
    type=["pdf", "txt", "docx", "csv"],
    accept_multiple_files=True,
    key="file_uploader"
)

# Handle new file uploads (avoid duplicates)
if uploaded:
    allowed_extensions = ['.pdf', '.txt', '.docx', '.csv']
    existing_names = [f.name for f in st.session_state.uploaded_files]
    new_files_added = False
    invalid_files = []
    
    for file in uploaded:
        file_ext = '.' + file.name.split('.')[-1].lower()
        if file_ext not in allowed_extensions:
            invalid_files.append(file.name)
        elif file.name not in existing_names:
            st.session_state.uploaded_files.append(file)
            existing_names.append(file.name)
            new_files_added = True
    
    if invalid_files:
        st.sidebar.error(f"❌ Invalid file format(s): {', '.join(invalid_files)}. Only PDF, TXT, DOCX, CSV files are allowed.")
    
    if new_files_added:
        st.rerun()

print("new file names",st.session_state.uploaded_files)

if len(st.session_state.uploaded_files) > 0:
    st.sidebar.success("✅ Files indexed!")

file_details = st.session_state.uploaded_files
load_files_status = load_files(file_details)

# Show uploaded files with delete option
if len(st.session_state.uploaded_files) == 0:
    st.sidebar.info("No files uploaded yet.")
else:
    files_to_remove = []
    for i, f in enumerate(st.session_state.uploaded_files):
        col1, col2 = st.sidebar.columns([3, 1])
        col1.write(f"📄 **{f.name}**")
        if col2.button("❌", key=f"del_{f.name}_{i}"):
            files_to_remove.append(f)
    
    # Remove files after iteration
    if files_to_remove:
        for file_to_remove in files_to_remove:
            if file_to_remove in st.session_state.uploaded_files:
                st.session_state.uploaded_files.remove(file_to_remove)
        st.rerun()

# Clear buttons
st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear All Files"):
    st.session_state.uploaded_files = []
    st.rerun()

if st.sidebar.button("💬 Clear Chat"):
    st.session_state.messages = []
    st.rerun()

st.sidebar.markdown("---")

# -----------------------------
# MAIN CHAT AREA
# -----------------------------
st.title("💬 Network Issue troubleshooter")


# Display chat history
for message in st.session_state.messages:
    if message["sender"] == "user":
        st.markdown(f"**You:** {message['text']}")
    else:
        st.markdown(f"**AI:** {message['text']}")

# -----------------------------
# USER INPUT BOX
# -----------------------------
st.markdown("---")
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Type your message…", key="input_msg")
    submit_button = st.form_submit_button("Send")

def displayAiResponse(botResponse):
    """
    Displays the AI response in the Streamlit chat interface
    and stores it in session_state for chat history.
    """
    # Append AI response to session state history
    st.session_state.messages.append({"sender": "ai", "text": botResponse})

def fetchAIResponse(user_input):
    # Check if any FAISS indices exist
    indices = ["faiss_index_pdf", "faiss_index_docx", "faiss_index_txt", "faiss_index_csv"]
    existing_indices = [idx for idx in indices if os.path.exists(f"{idx}.faiss") or os.path.exists(idx)]
    
    if not existing_indices:
        displayAiResponse("Please upload and process some files first before asking questions.")
        return
    
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Load existing vectorstores
    vectorstores = []
    for idx in existing_indices:
        try:
            vs = FAISS.load_local(idx, embeddings, allow_dangerous_deserialization=True)
            vectorstores.append(vs)
        except:
            continue
    
    if not vectorstores:
        displayAiResponse("No valid document indices found. Please upload files first.")
        return
    
    # Merge all vectorstores
    main_vectorstore = vectorstores[0]
    for vs in vectorstores[1:]:
        main_vectorstore.merge_from(vs)
    
    retriever = main_vectorstore.as_retriever()
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0), retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    res = retrieval_chain.invoke({"input": user_input})
    displayAiResponse(res['answer'])



# Handle form submission
if submit_button and user_input:
    # Check if files are uploaded
    if len(st.session_state.uploaded_files) == 0:
        st.error("❌ Please upload at least one file before asking questions.")
    else:
        # Validate user input
        is_valid, error_message = validate_user_input(user_input)
        
        if not is_valid:
            st.error(error_message)
        else:
            # Sanitize input
            sanitized_input = sanitize_input(user_input)
            
            # Check if network-related (optional warning)
            if not is_network_related(sanitized_input):
                st.warning("⚠️ This question doesn't seem network-related. For best results, ask about network troubleshooting.")
            
            # Add user message
            st.session_state.messages.append({"sender": "user", "text": sanitized_input})
            
            # Initialize LLM and tools
            tools = [format_gemini_response]
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, callbacks=[AgentCallbackHandler()])
            llm_with_tools = llm.bind_tools(tools)
            
            # Show typing indicator and get AI response
            with st.spinner("AI is thinking…"):
                fetchAIResponse(sanitized_input)
            
            st.rerun()