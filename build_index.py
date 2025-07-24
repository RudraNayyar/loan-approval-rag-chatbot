import os
import faiss
from sentence_transformers import SentenceTransformer

# Path to the chunks file
data_path = 'chunks.txt'
index_path = 'faiss_index.bin'
embeddings_path = 'embeddings.npy'

# Load text chunks
def load_chunks(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    # Split on separator
    chunks = [chunk.strip() for chunk in text.split('\n---\n') if chunk.strip()]
    print(f"Loaded {len(chunks)} chunks.")
    return chunks

# Embed chunks
def embed_chunks(chunks, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True)
    print(f"Generated {len(embeddings)} embeddings.")
    return embeddings

# Build FAISS index
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print(f"FAISS index built with {index.ntotal} vectors.")
    return index

if __name__ == "__main__":
    chunks = load_chunks(data_path)
    embeddings = embed_chunks(chunks)
    # Save embeddings for later use
    import numpy as np
    np.save(embeddings_path, embeddings)
    # Build and save FAISS index
    index = build_faiss_index(embeddings)
    faiss.write_index(index, index_path)
    print(f"Index and embeddings saved.")
