
from langchain.agents import create_tool_calling_agent,AgentExecutor,Tool
from langchain_ollama import ChatOllama
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

search_tool=Tool(
    name='duckduckgosearch',
    func=DuckDuckGoSearchRun().invoke,
    description='performs the web search'

)
prompt=ChatPromptTemplate.from_messages(
    [
        ('system','you are assistant that gives all answers'),
        ('human','{input}'),
        MessagesPlaceholder(variable_name='agent_scratchpad')
    ]
)

llm=ChatOllama(model='mistral')
tools=[
    Tool
    (
        name='search',
        func=search_tool.invoke,
        description='use this tool for perform search'
    )
]
agent=create_tool_calling_agent(llm,tools=tools,prompt=prompt)
agent_executor=AgentExecutor(agent=agent, tools=tools,verbose=True)
response=agent_executor.invoke({'input':'what is llm?'})
print(response.content)