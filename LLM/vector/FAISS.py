import faiss
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
docs=[
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning is a part of machine learning that uses neural networks.",
    "Neural networks are powerful models for image recognition.",
    "FAISS enables efficient similarity search on large datasets.",
    "Reinforcement learning is used in gaming and robotics."
]
embeddings = model.encode(docs)
print("Embedding shape:", embeddings.shape)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
query_text = input(" Enter your search query: ")
query_embedding = model.encode([query_text])
_, indices = index.search(query_embedding, k=2)

print("FAISS Result:", docs[indices[0][0]])