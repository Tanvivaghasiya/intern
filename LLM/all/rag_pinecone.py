from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
from dotenv import load_dotenv

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

prompt_template = PromptTemplate(
    template="""
Use the following context to answer the question:

Context: {context}

Question: {input}

If the answer is not found in give "ans not found."
""",
    input_variables=["input", 'context'],
)

loader = PyPDFLoader("ch-1.pdf")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=3)
split_docs = text_splitter.split_documents(docs)

embeddings = OllamaEmbeddings(model="mistral")
vector_store = PineconeVectorStore.from_documents(split_docs, embeddings, index_name="chatbot")

llm = ChatOllama(model="mistral")
retriever = vector_store.as_retriever()

question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
chain = create_retrieval_chain(retriever, question_answer_chain)


session_histories={}
def get_session_history(user_id:str,conversation_id:str):
    key = f"{user_id}_{conversation_id}"
    if key not in session_histories:
        session_histories[key] = InMemoryChatMessageHistory()
    session_histories[key].messages = session_histories[key].messages[-10:]

    return session_histories[key]

llm=ChatOllama(model='llama3')
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You're an helpful assistant",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)
runnable=prompt|llm

with_message_history=RunnableWithMessageHistory(
     runnable,
     get_session_history,
     input_messages_key='input',
     history_messages_key='history',
     history_factory_config=[
         ConfigurableFieldSpec(
             id='user_id',
             annotation='str',
             name='user_id',
             description='unique user id',
             default=" ",
             is_shared=True,
         ),
         ConfigurableFieldSpec(
             id="conversation_id",
             annotation="str",
             name="conversation_id",
             description='unique conversation id',
             default=" ",
             is_shared=True

         ),
     ],

 )

def ask_question(query: str) -> str:

    relevant_docs = retriever.invoke(query)
    if not relevant_docs:
        return "ans not found."

    result = with_message_history.invoke(
        {"input": query},
        config={"configurable": {"user_id": '123', "conversation_id": '1'}}
    )
    return result

while True:
    query = input("ask a question: ")
    if query.lower() == "exit":
        break
    answer = ask_question(query)
    print("ans:", answer)