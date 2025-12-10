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

if "microservices" not in st.session_state:
    st.session_state.microservices = []


# -----------------------------
# SIDEBAR — FILE UPLOADS
# -----------------------------
st.sidebar.title("📂 Uploaded Files")

# Hide default file display
st.sidebar.markdown("""
<style>
.uploadedFile, .stFileUploader > div > div > div > div:nth-child(2) {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

uploaded = st.sidebar.file_uploader(
    "Upload multiple files",
    type=["pdf", "txt", "docx", "csv", "xlsx", "xls"],
    accept_multiple_files=True,
    key="file_uploader"
)

# Handle new file uploads (avoid duplicates)
if uploaded:
    allowed_extensions = ['.pdf', '.txt', '.docx', '.csv', '.xlsx', '.xls']
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
        st.sidebar.error(f"❌ Invalid file format(s): {', '.join(invalid_files)}. Only PDF, TXT, DOCX, CSV, Excel files are allowed.")
    
    if new_files_added:
        st.rerun()

# Single file status display
if len(st.session_state.uploaded_files) > 0:
    st.sidebar.success(f"✅ {len(st.session_state.uploaded_files)} file(s) indexed!")
    
    # Process files
    file_details = st.session_state.uploaded_files
    load_files_status = load_files(file_details)
else:
    st.sidebar.info("No files uploaded yet.")

# Clear buttons
st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear All Files"):
    st.session_state.uploaded_files = []
    st.rerun()

if st.sidebar.button("💬 Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# -----------------------------
# MICROSERVICE LOGS SECTION
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.title("🔍 Microservice Logs")

# Initialize microservice list in session state
if "microservices" not in st.session_state:
    st.session_state.microservices = []

# Input for new microservice
new_microservice = st.sidebar.text_input("Add microservice name:", placeholder="e.g., auth-service")
if st.sidebar.button("➕ Add Service") and new_microservice:
    if new_microservice not in st.session_state.microservices:
        st.session_state.microservices.append(new_microservice)
        st.rerun()

# Display added microservices
if st.session_state.microservices:
    st.sidebar.success(f"📋 {len(st.session_state.microservices)} service(s) configured")
    
    services_to_remove = []
    for i, service in enumerate(st.session_state.microservices):
        col1, col2 = st.sidebar.columns([3, 1])
        col1.write(f"🔧 **{service}**")
        if col2.button("❌", key=f"del_service_{i}"):
            services_to_remove.append(service)
    
    # Remove services
    if services_to_remove:
        for service in services_to_remove:
            st.session_state.microservices.remove(service)
        st.rerun()
        
    # Clear all services button
    if st.sidebar.button("🗑️ Clear All Services"):
        st.session_state.microservices = []
        st.rerun()
else:
    st.sidebar.info("No microservices configured yet.")

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

# -----------------------------
# MICROSERVICE LOG FUNCTIONS
# -----------------------------
def fetch_microservice_logs(service_name, lines=100, time_range="1h"):
    """Fetch logs from a specific microservice."""
    try:
        import subprocess
        import json
        from datetime import datetime, timedelta
        
        # Calculate time range
        end_time = datetime.now()
        if time_range == "1h":
            start_time = end_time - timedelta(hours=1)
        elif time_range == "24h":
            start_time = end_time - timedelta(hours=24)
        else:
            start_time = end_time - timedelta(hours=1)
        
        # Try kubectl logs first (Kubernetes)
        try:
            cmd = f"kubectl logs -l app={service_name} --tail={lines} --since={time_range}"
            result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and result.stdout:
                return {
                    "status": "success",
                    "logs": result.stdout,
                    "source": "kubernetes",
                    "service": service_name,
                    "lines": len(result.stdout.split('\n'))
                }
        except Exception:
            pass
        
        # Try docker logs (Docker)
        try:
            cmd = f"docker logs {service_name} --tail {lines}"
            result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and result.stdout:
                return {
                    "status": "success",
                    "logs": result.stdout,
                    "source": "docker",
                    "service": service_name,
                    "lines": len(result.stdout.split('\n'))
                }
        except Exception:
            pass
        
        # Fallback: simulate logs for demo
        sample_logs = f"""[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INFO  {service_name} - Service started successfully
[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] DEBUG {service_name} - Processing request ID: req-12345
[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] WARN  {service_name} - High memory usage detected: 85%
[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INFO  {service_name} - Request completed in 250ms
[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR {service_name} - Connection timeout to database"""
        
        return {
            "status": "success",
            "logs": sample_logs,
            "source": "demo",
            "service": service_name,
            "lines": len(sample_logs.split('\n'))
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "service": service_name
        }

def fetch_all_microservice_logs(service_list, lines=100):
    """Fetch logs from multiple microservices."""
    all_logs = {}
    for service in service_list:
        log_result = fetch_microservice_logs(service, lines)
        all_logs[service] = log_result
    return all_logs

def format_log_response(log_data):
    """Format log data for display in chatbot."""
    if not log_data:
        return "No microservices configured for log fetching."
    
    response = "**Microservice Logs:**\n\n"
    
    for service, data in log_data.items():
        if data["status"] == "success":
            response += f"**🔧 {service}** ({data['source']}) - {data['lines']} lines:\n"
            response += f"```\n{data['logs'][:1000]}{'...' if len(data['logs']) > 1000 else ''}\n```\n\n"
        else:
            response += f"**❌ {service}** - Error: {data.get('error', 'Unknown error')}\n\n"
    
    return response

def render_vm_chart(query_result):
    """Render chart from VM query result data in both chatbot and terminal."""
    try:
        import plotly.graph_objects as go
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import datetime
        
        result_data = query_result["data"]["result"]
        if result_data:
            # Create Plotly figure for Streamlit
            fig = go.Figure()
            
            # Create matplotlib figure for terminal
            plt.figure(figsize=(12, 6))
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            
            for i, series in enumerate(result_data):
                if "values" in series and series["values"]:
                    timestamps = [float(val[0]) for val in series["values"]]
                    values = [float(val[1]) for val in series["values"]]
                    dates = [datetime.datetime.fromtimestamp(ts) for ts in timestamps]
                    
                    metric_info = series.get("metric", {})
                    metric_name = metric_info.get("__name__", f"Series {i+1}")
                    
                    labels = []
                    for k, v in metric_info.items():
                        if k != "__name__" and len(labels) < 2:
                            labels.append(f"{k}={v}")
                    
                    if labels:
                        metric_name += f" ({', '.join(labels)})"
                    
                    color = colors[i % len(colors)]
                    
                    # Add to Plotly figure
                    fig.add_trace(go.Scatter(
                        x=dates, y=values, mode='lines+markers', name=metric_name,
                        line=dict(color=color, width=3),
                        marker=dict(color=color, size=6, symbol='circle'),
                        hovertemplate='<b>%{fullData.name}</b><br>Time: %{x}<br>Value: %{y:.2f}<br><extra></extra>'
                    ))
                    
                    # Add to matplotlib figure
                    plt.plot(dates, values, color=color, linewidth=2, marker='o', markersize=4, label=metric_name)
            
            # Configure Plotly layout
            fig.update_layout(
                title=dict(text="📊 Victoria Metrics Time Series Analysis", x=0.5, font=dict(size=18, color='#2c3e50')),
                xaxis=dict(title=dict(text="Time (UTC)", font=dict(size=14, color='#34495e')), tickfont=dict(size=12), gridcolor='#ecf0f1', showgrid=True),
                yaxis=dict(title=dict(text="Metric Value", font=dict(size=14, color='#34495e')), tickfont=dict(size=12), gridcolor='#ecf0f1', showgrid=True),
                height=500, hovermode='x unified', showlegend=True,
                legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02, bgcolor="rgba(255,255,255,0.8)", bordercolor="#bdc3c7", borderwidth=1),
                plot_bgcolor='white', paper_bgcolor='#f8f9fa', margin=dict(l=60, r=120, t=80, b=60)
            )
            
            # Display in Streamlit chatbot
            st.plotly_chart(fig, use_container_width=True)
            
            # Configure and display matplotlib plot in terminal
            plt.title('Victoria Metrics - Time Series Analysis', fontsize=14, fontweight='bold')
            plt.xlabel('Time', fontsize=12)
            plt.ylabel('Metric Value', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            print("\n📊 Displaying plot in terminal...")
            plt.show()
            
            # Also try browser display
            try:
                print("📊 Opening interactive plot in browser...")
                fig.show(renderer="browser")
            except Exception as browser_error:
                print(f"Browser display failed: {browser_error}")
            
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        print(f"Chart error: {str(e)}")

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
    # Check if user input contains microservice log keywords
    log_keywords = ['logs', 'log', 'microservice', 'service logs', 'show logs', 'fetch logs']
    if any(keyword in user_input.lower() for keyword in log_keywords) and st.session_state.microservices:
        try:
            # Fetch logs from configured microservices
            log_data = fetch_all_microservice_logs(st.session_state.microservices)
            formatted_logs = format_log_response(log_data)
            displayAiResponse(formatted_logs)
            return
        except Exception as e:
            displayAiResponse(f"Error fetching microservice logs: {str(e)}")
            return
    
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
            
            # Display results with plot and summary
            summary_text = "\n".join([result.get('summary', '') for result in vm_results if result.get('summary')])
            response_text = f"**Victoria Metrics Query Results:**\n\n{vm_output}\n\n**Summary of Changes:**\n{summary_text}"
            displayAiResponse(response_text, plot_data)
            
            # Chart will be rendered by displayAiResponse function
            
            return
        except Exception as e:
            displayAiResponse(f"Error processing VM query: {str(e)}")
            return
    
    # Check if any FAISS indices exist
    indices = ["faiss_index_pdf", "faiss_index_docx", "faiss_index_txt", "faiss_index_csv", "faiss_index_excel"]
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
    
    # Enhanced retriever with better parameters for CSV data
    retriever = main_vectorstore.as_retriever(
        search_type="similarity",  # Use similarity for complete coverage
        search_kwargs={"k": 20, "fetch_k": 40}  # Retrieve many chunks for complete PM lists
    )
    
    # Enhanced prompt for CSV and structured data
    custom_prompt = PromptTemplate.from_template(
        """You are a network troubleshooting expert. Use the following context to answer the question accurately.
        
        Context: {context}
        
        Question: {input}
        
        Instructions:
        1. Answer based ONLY on the provided context
        2. When asked for PM or KPI or TCE counters or lists, provide COMPLETE and COMPREHENSIVE lists - include ALL items found in the context
        3. For 5G network PM or KPI or TCE counters, list ALL counters with their full names and descriptions if available
        4. Present data in structured format using bullet points, numbered lists, or markdown tables
        5. Include ALL column headers and organize data logically - do NOT truncate or summarize lists
        6. If the list is long, still provide the COMPLETE list - users need all information
        7. For troubleshooting queries, provide specific, actionable steps
        8. Maintain data structure from CSV/Excel files exactly as provided
        9. If context doesn't contain relevant information, say "I don't have enough information in the uploaded documents to answer this question."
        
        Answer:"""
    )
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.1,
        max_output_tokens=8192  # Increase token limit for complete responses
    )
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
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1, callbacks=[AgentCallbackHandler()])
            llm_with_tools = llm.bind_tools(tools)
            
            # Show typing indicator and get AI response
            with st.spinner("AI is thinking…"):
                fetchAIResponse(sanitized_input)
            
            st.rerun()