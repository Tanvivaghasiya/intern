from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

loader=PyPDFLoader('Weekly Diaries.pdf')
docs=loader.load()
print(docs)
splitter=RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=20)
chunk=splitter.split_documents(docs)
print(chunk)
embedding=OllamaEmbeddings(model='llama3')
db=FAISS.from_documents(chunk,embedding)
print(db)
prompt=ChatPromptTemplate.from_template('''
ans the following que based on the provided document
<context>
{context}
<context>
Question:{input}
'''
)
llm=Ollama(model='llama3')
print(llm)
doc_chain=create_stuff_documents_chain(llm, prompt)
print(doc_chain)
retriever=db.as_retriever()
print(retriever)

retrieval_chain=create_retrieval_chain(retriever,doc_chain)
print(retrieval_chain)
response=retrieval_chain.invoke({'input':'types of ai'})
print(response)