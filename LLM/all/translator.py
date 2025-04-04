from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

prompt=ChatPromptTemplate.from_template(
    """ Translate the text from {source_language} to {target_language}
    text : {text}
    Translated text:"""

)
llm=ChatOllama(model='llama3')
chain=prompt|llm
source_language=input('enter the source language:')
target_language=input('enter the target language:')
text=input('enter the text:')
translated_text=chain.invoke({
    'target_language':target_language,
    'source_language':source_language,
    'text':text})
print('translated text:',translated_text.content)