from langchain.document_loaders import PyMuPDFLoader
loader=PyMuPDFLoader("Weekly Diaries.pdf")
doc=loader.load()
for docc in doc:
    print((docc.page_content))

#word
from langchain.document_loaders import UnstructuredWordDocumentLoader
loader = UnstructuredWordDocumentLoader("Diary.docx")
document = loader.load()
for doc in document:
    print(doc.page_content)
