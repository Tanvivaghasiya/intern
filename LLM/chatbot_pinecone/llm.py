import os
import streamlit as st
import ollama
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "pcsk_6fBJ7B_DbPgeQVRwkA9dL237CHmVDHk5nsSMzhPnaCa1w1PXy9Mz8yR56NXm22CJjh5ZcL")  # Replace if not using env vars
index_name = "chatbot-index"

pc = Pinecone(api_key=PINECONE_API_KEY)

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(index_name)

embeddings = OllamaEmbeddings(model="mistral")

vector_store = PineconeVectorStore(index, embeddings, text_key="text")
st.title("Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

query = st.chat_input("Ask que:")

if query:
    st.chat_message("user").write(query)
    docs = vector_store.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}]
    )
    bot_reply = response["message"]["content"]

    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    st.chat_message("assistant").write(bot_reply)
