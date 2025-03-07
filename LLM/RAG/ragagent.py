from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool

from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.agents import create_openai_tools_agent,AgentExecutor

apiwrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=100)
wiki=WikipediaQueryRun(api_wrapper=apiwrapper)
print(wiki.name)
loader=WebBaseLoader("https://www.promptingguide.ai/techniques/fewshot")
docs=loader.load()
documents=RecursiveCharacterTextSplitter(chunk_size=150,chunk_overlap=10).split_documents(docs)
embedding=OllamaEmbeddings(model='mistral')
db=FAISS.from_documents(documents,embedding)
retriever=db.as_retriever
print(retriever)
retrieval_tool=create_retriever_tool(retriever,'search tool',"retrieves docs from the vectorstore")
print(retrieval_tool.name)

arxiv_wrapper=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=150)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)
print(arxiv.name)


tools=[wiki,arxiv,retrieval_tool]
print(tools)
llm=ChatOllama(model='mistral',temperature=0.2)
System_prompt=ChatPromptTemplate(
    [
        ('system','you are helpful assistant'),
        ('human','Question:{input}'),
        MessagesPlaceholder(variable_name='agent_scratchpad')
    ]
)
agent=create_openai_tools_agent(llm,tools,System_prompt)
print(agent)
agent_executor=AgentExecutor(agent=agent,tools=tools,verbose=True)
print(agent_executor)
response=agent.invoke({'input':'what is prompt','intermediate_steps':[]})
print(response)