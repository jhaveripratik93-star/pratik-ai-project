import streamlit as st
import time
import base64
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langchain.tools import tool, BaseTool
from callbacks import AgentCallbackHandler

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

uploaded = st.sidebar.file_uploader(
    "Upload multiple files",
    type=["pdf", "txt", "docx", "xlsx", "csv", "json", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

if uploaded:
    for file in uploaded:
        st.session_state.uploaded_files.append(file)

# Show uploaded files
if len(st.session_state.uploaded_files) == 0:
    st.sidebar.info("No files uploaded yet.")
else:
    for f in st.session_state.uploaded_files:
        st.sidebar.write(f"📄 **{f.name}**")

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
    # Start conversation    
    messages = [HumanMessage(content = {user_input})]
    while True:
        ai_message = llm_with_tools.invoke(messages)

        # If the model decides to call tools, execute them and return results
        tool_calls = getattr(ai_message, "tool_calls", None) or []
        if len(tool_calls) > 0:
            messages.append(ai_message)
            for tool_call in tool_calls:
                # tool_call is typically a dict with keys: id, type, name, args
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                tool_call_id = tool_call.get("id")

                tool_to_use = find_tool_by_name(tools, tool_name)
                observation = tool_to_use.invoke(tool_args)
                print(f"observation={observation}")

                messages.append(
                    ToolMessage(content=str(observation), tool_call_id=tool_call_id)
                )
            # Continue loop to allow the model to use the observations
            continue

        # No tool calls -> final answer
        output = ai_message.content
        print("message to be shown the user")
        print(output[0]['text'])
        displayAiResponse(output[0]['text'])
        break

# Handle form submission
if submit_button and user_input:
    # Add user message
    st.session_state.messages.append({"sender": "user", "text": user_input})
    
    # Initialize LLM and tools
    tools = [format_gemini_response]
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, callbacks=[AgentCallbackHandler()])
    llm_with_tools = llm.bind_tools(tools)
    
    # Show typing indicator and get AI response
    with st.spinner("AI is thinking…"):
        fetchAIResponse(user_input)
    
    st.rerun()