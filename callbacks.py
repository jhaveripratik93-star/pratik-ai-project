from langchain_core.callbacks import BaseCallbackHandler

class AgentCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print("LLM started...")

    def on_llm_end(self, response, **kwargs):
        print("LLM finished.")

    def on_tool_start(self, serialized, input_str, **kwargs):
        print(f"Tool started: {serialized.get('name')}")

    def on_tool_end(self, output, **kwargs):
        print(f"Tool output: {output}")
