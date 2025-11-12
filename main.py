import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import google.generativeai as genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

def main():
    print("Hellow world langchain code")
    information = '''
    my name is pratik
    I m working on creating ai model for esoa chatbot
    I would be using langchain
    chatbot would have functionality to fetch query response from spi doc using RAG
    chatbot would be able to connect to vm to fetch certain metric information
    '''
    summary_template = '''
    given the information {information} about the task generate
    1. A short summary 
    2. Important task
    '''
    summary_prompt_template = PromptTemplate(input_variables=["information"], 
    template = summary_template)

    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = summary_prompt_template.format(information=information)
    
    response = model.generate_content(prompt)

    answer = response.text
    print(prompt)
    print("#################################################")
    print(answer)



if __name__ == "__main__":
    main()