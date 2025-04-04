from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

prompt=ChatPromptTemplate.from_template("""
    generate the content for {content_type} on {topic}
    give the correct content
    """
)
llm=ChatOllama(model='llama3')
chain=prompt|llm

content_type=input('enter the type of content e.g. e-mail,articles,stories,marketing:')
topic=input('enter the topic:')

generate_content=chain.invoke({
    'content_type':content_type,'topic':topic
})
print(generate_content)
