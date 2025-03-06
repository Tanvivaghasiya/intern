from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

loader=PyMuPDFLoader("Weekly Diaries.pdf")
document=loader.load()
for doc in document:
    print(doc.page_content)

splitter=RecursiveCharacterTextSplitter(
    chunk_size=60,
    chunk_overlap=20
)
chunks = splitter.split_documents(doc)
print('len of chunks',len(chunks))
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(chunks, embeddings)
vector_store.save_local("faiss_index")

vector_store = FAISS.load_local("faiss_index", embeddings)

# Create retriever
retriever = vector_store.as_retriever()

# Initialize LLM (GPT-4)
llm = ChatOpenAI(model_name="gpt-4", temperature=0.5)

# Create RAG Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

# User Query
query = "What is the main topic of the document?"
response = qa_chain.run(query)

print(response)
