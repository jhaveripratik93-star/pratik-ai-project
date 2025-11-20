import streamlit as st
from dotenv import load_dotenv
from utils.loader import extract_text
from utils.safety import is_safe
#from utils.rag_pipeline import build_retriever, build_qa_chain
from utils.rag_pipeline import search
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List
from pydantic import BaseModel, Field
from langchain_community.tools.tavily_search import TavilySearchResults
#from langchain_tavily import TavilySearch
#from langchain_tavily import TravilySearch
# from langchain_core.output_parsers.pydantic import PydanticOutputParser
# from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables import RunnableLambda
from langchain.agents import create_agent
from utils.prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from utils.schemas import AgentResponse

#Load Environment variables
load_dotenv()


react_prompt_with_format_instructions = PromptTemplate(template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS, 
input_variables = ["input", "agent_scratchpad", "tool_names"]).partial(format_instructions=output_parser.get_format_instructions())

class source(BaseModel):
    """Schema for a source used by the agent"""
    source: str = Field(description="Source of the answer")

class AgentResponse(BaseModel):
    """Schema for the agent response with answers and sources"""
    output: str = Field(description="Final answer to the question")
    source: List[str] = Field(description="List of sources used to answer the question")


st.set_page_config(page_title="Doc Chatbot", layout="wide", page_icon="🤖")
st.title("🤖 Network trouble shooter")

st.sidebar.header("📂 Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs, DOCX, CSV, TXT",
    type=["pdf", "docx", "csv", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("Processing documents..."):
        all_texts = [extract_text(f) for f in uploaded_files]
        #print(all_texts)
#         retriever = build_retriever(all_texts)
#         qa_chain = build_qa_chain(retriever)
    st.sidebar.success("✅ Documents indexed!")
# else:
#     qa_chain = None
query = st.text_input("Ask a question about your uploaded documents:")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4)
tools = [TavilySearchResults()]
agent = create_agent(llm, tools, response_format=AgentResponse)


result = agent.invoke({"messages": [{
    "role": "user",
    "content": query
}]})

print(result["structured_response"])

#agent = create_agent(llm, tools, response_format=AgentResponse)


answer=''
if st.button("Generate Answer"):
    # if qa_chain:
    #     st.warning("Please upload documents first.")
    if not uploaded_files:
        st.warning("Please upload documents first.")
    elif not query.strip():
        st.warning("Enter a question.")
    else:
        with st.spinner("Fetching answer..."):
            try:
#                main(query)
#                res = qa_chain.invoke({"input": query})
                #print(query)
#                result = chain.invoke({"input": query})
#                print("result", result)
#                response = agent.invoke({"messages":HumanMessage(content={query})})
                #print(response)
                #print(response['messages'])
                 

#                tool_message = response['messages'][-1]  # Last AI message that has the answer
#                answer = tool_message['tool_calls'][0]['args']['output']  # Extract the 'output' field from the first tool call

                #answer = res.get("output", "")
                #print(answer)
#                answer = res
            except Exception as e:
                st.error(f"Error generating response: {e}")
                answer = None

        if answer:
            if not is_safe(query):
                st.error("⚠️ Unsafe content detected.")
            else:
                st.markdown("### 🤖 AI Proposed Answer")
                st.write(answer)

                # st.markdown("---\n### 🧍 Human Review")
                # human_edit = st.text_area("Edit or approve:", value=answer)
                # if st.button("Submit Final Answer"):
                #     st.success("✅ Final answer approved:")
                #     st.write(human_edit)
        else:
            st.warning("No answer generated — try rephrasing your question.")

