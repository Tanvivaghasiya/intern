from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
prompt=ChatPromptTemplate.from_messages(
    [
        ('system','you are helpful assistant that answers all the questions'),
        ('human','{input}')
    ]
)
llm=OllamaLLM(model='deepseek-r1')
chain=prompt|llm

response=chain.invoke({'input':'what is llm?'})
print(response)

