from langchain.document_loaders import TextLoader
loader = TextLoader("llm.txt")
documents = loader.load()
for doc in documents:
    print(doc.page_content)

#csv
from langchain.document_loaders import CSVLoader
loader=CSVLoader("ecommerce.csv")
doc=loader.load()
for docc in doc:
    print(docc.page_content)