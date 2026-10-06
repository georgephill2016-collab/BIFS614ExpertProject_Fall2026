# build_index.py

from sentence_transformers import SentenceTransformer
import faiss
import pickle

# Load document
with open("knowledge.txt", "r", encoding="utf-8") as f:
    text = f.read()

# commented out chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
# Split by chunks
def chunk_text(text, chunk_size=150, overlap=60):
    """
    Split a document into overlapping chunks by word count.
    """
    words = text.split()
    chunks = []

    step = chunk_size - overlap

    for start in range(0, len(words), step):
        chunk = words[start:start + chunk_size]

        if chunk:
            chunks.append(" ".join(chunk))

        if start + chunk_size >= len(words):
            break

    return chunks

chunks = chunk_text(text)

# Embedding model
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# Convert chunks to vectors
embeddings = embedding_model.encode(chunks, normalize_embeddings=True)

# Create FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save index
faiss.write_index(index, "knowledge.index")

# Save chunks
with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("Index created successfully.")
