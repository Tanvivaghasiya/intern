from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory,BaseChatMessageHistory
from langchain_ollama import ChatOllama
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

session={}
def get_session_history(user_id:str,conversation_id:str) -> BaseChatMessageHistory:
    if (user_id,conversation_id) not in session:
        session[(user_id,conversation_id)]=InMemoryChatMessageHistory()
    return session[(user_id,conversation_id)]

prompt=ChatPromptTemplate.from_messages(
    [
        ('system','you are helpful assistant'),
        MessagesPlaceholder(variable_name='history'),
        ('human','{input}')
    ]
)

llm=ChatOllama(model='llama3')
runnable=prompt|llm

with_message_history=RunnableWithMessageHistory(
    runnable,
    get_session_history=get_session_history,
    input_messages_key='input',
    history_messages_key='history',
    history_factory_config=[
        ConfigurableFieldSpec(
        id='user_id',
        annotation=str,
        name='user_id',
        description='unique user id',
        default=" ",
        is_shared=True

    ),
    ConfigurableFieldSpec(
        id='conversation_id',
        annotation=str,
        name='conversation_id',
        description='unique conversation id',
        is_shared=True,
        default=" "

),
],
)
while True:
    user_input=input('user:')
    if user_input=='exit':
        break

    result=with_message_history.invoke(
        {'input':user_input},
        config={'configurable':{'user_id':'123','conversation_id':"2"}}
    )
    print('ans',result.content)