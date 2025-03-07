from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

loader=TextLoader("speech.txt")
docs=loader.load()
print(docs)
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
documents=text_splitter.split_documents(docs)
print(documents)
embeddings=OllamaEmbeddings(model="llama3")
db = FAISS.from_documents(documents,embeddings)
retriever = db.as_retriever()

print(db)
query="the encoder is composed of 6 identical layers"
result=db.similarity_search(query)
(print(result[0].page_content))

llm = OllamaLLM(model="llama3", temperature=0.5)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

query = "What is the main topic?"
response = qa_chain.invoke(query)

print(response)
