from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.chat_models import ChatOpenAI

loader=PyPDFLoader("Weekly Diaries.pdf")
document=loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(document)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embedding_model)
vector_store.save_local("my_vector_store")
retriever = vector_store.as_retriever()

while True:
    query = input("\nAsk a question (or type 'exit' to quit): ")
    if query.lower() == "exit":
        break
    docs = retriever.get_relevant_documents(query)
    print("\nAnswer:", docs[0].page_content if docs else "No relevant information found.")