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
from langchain_core.prompts import PromptTemplate


load_dotenv()

import json
import re
import plotly.graph_objects as go
import streamlit as st

def extract_code_blocks(text):
    """Extract all ``` blocks and inline fig = {...} definitions."""
    blocks = []

    # Capture fenced blocks
    fenced = re.findall(r"```(?:json|python|js|plotly)?\s*(.*?)```", text, flags=re.DOTALL)
    blocks.extend(fenced)

    # Capture inline `fig = {...}`
    inline = re.findall(r"fig\s*=\s*(\{.*?\})", text, flags=re.DOTALL)
    blocks.extend(inline)

    return blocks


def js_to_json(clean_js):
    """Convert JS-like Plotly dict → JSON-friendly Python dict."""
    return (
        clean_js.replace("'", '"')
                .replace("True", "true")
                .replace("False", "false")
                .replace("None", "null")
                .replace("new Date", "")
                .replace("toLocaleString()", "")
                .replace(";", "")
    )


def safe_json_loads(text):
    """Load JSON or fallback to eval() with protection."""
    try:
        return json.loads(text)
    except Exception:
        try:
            return eval(text, {"__builtins__": None}, {})
        except Exception:
            return None


def build_figure_from_dict(fig_dict, df=None):
    """Convert parsed JSON/Python dict into a Plotly Figure."""
    fig = go.Figure()

    traces = fig_dict.get("data", [])

    for tr in traces:
        # Use VM data if provided
        if df is not None:
            x_vals = [v[0] for v in df]
            y_vals = [v[1] for v in df]
        else:
            x_vals = tr.get("x", [])
            y_vals = tr.get("y", [])

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode=tr.get("mode", "lines"),
                name=tr.get("name", "Series")
            )
        )

    fig.update_layout(**fig_dict.get("layout", {}))
    return fig


def render_plotly_from_ai_text(ai_text, df=None):
    """
    Detect multiple Plotly snippets in AI response,
    convert them to Python, and render inside Streamlit.
    Returns True if at least 1 plot was rendered.
    """
    blocks = extract_code_blocks(ai_text)

    if not blocks:
        return False

    figs = []

    for raw_block in blocks:
        cleaned = js_to_json(raw_block.strip())
        fig_dict = safe_json_loads(cleaned)

        if not fig_dict or "data" not in fig_dict:
            continue

        figs.append(build_figure_from_dict(fig_dict, df=df))

    if not figs:
        return False

    # If multiple charts → Tab UI
    if len(figs) > 1:
        tabs = st.tabs([f"Chart {i+1}" for i in range(len(figs))])
        for tab, chart in zip(tabs, figs):
            with tab:
                st.plotly_chart(chart, use_container_width=True)
    else:
        st.plotly_chart(figs[0], use_container_width=True)

    return True

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

if "vm_history" not in st.session_state:
    st.session_state.vm_history = []


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

def render_vm_chart(query_result):
    """Render Plotly chart from VM query result data."""
    try:
        import plotly.graph_objects as go
        import datetime
        
        result_data = query_result["data"]["result"]
        if result_data:
            fig = go.Figure()
            
            for series in result_data:
                if "values" in series:
                    timestamps = [float(val[0]) for val in series["values"]]
                    values = [float(val[1]) for val in series["values"]]
                    dates = [datetime.datetime.fromtimestamp(ts) for ts in timestamps]
                    
                    metric_name = series.get("metric", {}).get("__name__", "Unknown Metric")
                    fig.add_trace(go.Scatter(
                        x=dates,
                        y=values,
                        mode='lines+markers',
                        name=metric_name
                    ))
            
            fig.update_layout(
                title="Victoria Metrics Query Results",
                xaxis_title="Time",
                yaxis_title="Value",
                height=400
            )
            
            st.plotly_chart(fig, width=True)
            
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")

def displayAiResponse(botResponse, plot_data=None):
    """Display AI response and optional chart."""
    st.session_state.messages.append({
        "sender": "ai",
        "text": botResponse,
        "plot_data": plot_data
    })
    
    st.markdown(f"**AI:** {botResponse}")
    
    # Render VM chart if available
    if plot_data and plot_data.get("query_result", {}).get("status") == "success":
        render_vm_chart(plot_data["query_result"])


# def displayAiResponse(botResponse, plot_data=None):
#     """
#     Displays the AI response in the Streamlit chat interface
#     and stores it in session_state for chat history.
#     """
#     # Append AI response to session state history
#     st.session_state.messages.append({"sender": "ai", "text": botResponse, "plot_data": plot_data})

def fetchAIResponse(user_input):
    # Check if user input contains VM or Victoria Metrics keywords
    vm_keywords = ['vm', 'victoria metrics', 'victoriametrics', 'metrics', 'monitoring']
    if any(keyword in user_input.lower() for keyword in vm_keywords):
        try:
            import io
            import sys
            import json
            from vm_dbconnection import vm_dbconnectionMain, run_metrics_query, generate_plotly_code_with_groq, generate_metrics_ql_query
            
            # Get MetricsQL query and run it
            metrics_ql_query = generate_metrics_ql_query(user_input)
            query_result = run_metrics_query(metrics_ql_query, time_range="1h", step="5m")
            
            # Generate plot code
            plotly_code = generate_plotly_code_with_groq(json.dumps(query_result.get("data", {})), user_input)
            
            # Capture text output and get results with history
            old_stdout = sys.stdout
            sys.stdout = captured_output = io.StringIO()
            
            # Import the modified function
            from vm_dbconnection import vm_dbconnectionMain
            vm_results = vm_dbconnectionMain([user_input], st.session_state.vm_history)
            
            sys.stdout = old_stdout
            vm_output = captured_output.getvalue()
            
            # Store results in history
            if vm_results:
                st.session_state.vm_history.extend(vm_results)
                # Keep only last 10 entries
                st.session_state.vm_history = st.session_state.vm_history[-10:]
            
            # Prepare plot data if valid
            plot_data = None
            if plotly_code and plotly_code != "NO_PLOT":
                plot_data = {
                    "query_result": query_result,
                    "plotly_code": plotly_code
                }
            
            # Display results with plot
            response_text = f"**Victoria Metrics Query Results:**\n\n{vm_output}"
            displayAiResponse(response_text, plot_data)
            
            # Chart will be rendered by displayAiResponse function
            
            return
        except Exception as e:
            displayAiResponse(f"Error processing VM query: {str(e)}")
            return
    
    # Check if any FAISS indices exist
    indices = ["faiss_index_pdf", "faiss_index_docx", "faiss_index_txt", "faiss_index_csv"]
    existing_indices = [idx for idx in indices if os.path.exists(f"{idx}.faiss") or os.path.exists(idx)]
    
    if not existing_indices:
        displayAiResponse("Please upload and process some files first before asking questions.")
        return
    
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.prompts import PromptTemplate
    
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
    
    # Enhanced retriever with better parameters
    retriever = main_vectorstore.as_retriever(
        search_type="mmr",  # Maximum Marginal Relevance for diversity
        search_kwargs={"k": 5, "fetch_k": 10}  # Retrieve more relevant chunks
    )
    
    # Custom prompt for better accuracy
    custom_prompt = PromptTemplate.from_template(
        """You are a network troubleshooting expert. Use the following context to answer the question accurately.
        Search for the answer in all the uploaded documents. Give the best combined result for the user query
        
        Context: {context}
        
        Question: {input}
        
        Instructions:
        1. Answer based ONLY on the provided context
        2. If the context doesn't contain relevant information, say "I don't have enough information in the uploaded documents to answer this question."
        3. Provide specific, actionable troubleshooting steps when possible
        4. Include relevant technical details from the context
        5. Be concise but comprehensive
        
        Answer:"""
    )
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)  # Slightly higher temp for better responses
    combine_docs_chain = create_stuff_documents_chain(llm, custom_prompt)
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
        elif user_input.lower() == "exit":
            st.session_state.messages.append({"sender": "ai", "text": "Goodbye!"})
        elif user_input.lower() == "show files":
            st.session_state.messages.append({"sender": "ai", "text": "Uploaded files: " + ", ".join([f.name for f in st.session_state.uploaded_files])})
        elif user_input.lower() == "show history":
            st.session_state.messages.append({"sender": "ai", "text": "Chat history: " + ", ".join([msg['text'] for msg in st.session_state.messages])})
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
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, callbacks=[AgentCallbackHandler()])
            llm_with_tools = llm.bind_tools(tools)
            
            # Show typing indicator and get AI response
            with st.spinner("AI is thinking…"):
                fetchAIResponse(sanitized_input)
            
            st.rerun()