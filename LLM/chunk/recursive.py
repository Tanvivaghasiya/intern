from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

docs = TextLoader("llm.txt").load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

for doc in docs:
    chunks = splitter.split_text(doc.page_content)
    print("\n---\n".join(chunks))
