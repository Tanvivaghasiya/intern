import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

st.title('Content Generator')

content_type=st.text_input('enter the type of content e.g. e-mail,articles,stories,marketing:')
topic=st.text_input('enter the topic:')

if st.button('generate content'):
    prompt=ChatPromptTemplate.from_template("""
    generate the content for {content_type} on {topic}
    give the correct content
    """
)
    llm=ChatOllama(model='llama3')
    chain=prompt|llm


    generate_content=chain.invoke({
    'content_type':content_type,'topic':topic
})
    st.subheader("Generated Content:")
    st.write(generate_content.content)
