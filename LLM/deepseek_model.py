from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import streamlit as st
st.title("chatbot[Deepseek/langchain]")
template = """Question: {question}
Answer: Let's think step by step."""
prompt = ChatPromptTemplate.from_template(template)
model = OllamaLLM(model="deepseek-r1")
chain = prompt | model
question = st.chat_input("Ask the question")
if question:
    st.write(chain.invoke({"question": question}))