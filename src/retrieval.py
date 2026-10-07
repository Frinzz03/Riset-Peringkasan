import numpy as np
import config
from src.chunking_indexing import get_ollama_embedding, load_index

def cosine_similarity(query_vec: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    """Menghitung cosine similarity antara query vector dan seluruh embeddings."""
    # Normalisasi vektor
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    emb_norms = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
    # Cosine similarity = dot product dari vektor yang sudah dinormalisasi
    similarities = np.dot(emb_norms, query_norm)
    return similarities

def retrieve_top_k(query: str, top_k: int = 5, embed_model: str = config.DEFAULT_EMBED_MODEL) -> list:
    """
    Melakukan similarity search pada index vektor lokal untuk mengambil Top-K chunk teratas.
    """
    # Muat index dari disk
    embeddings, metadata = load_index()
    
    # Generate query embedding
    query_vec = get_ollama_embedding(query, model_name=embed_model)
    if not query_vec:
        raise ValueError("Gagal membuat embedding untuk query.")
    
    query_vec = np.array(query_vec, dtype=np.float32)
    
    # Hitung cosine similarity
    similarities = cosine_similarity(query_vec, embeddings)
    
    # Ambil Top-K indeks dengan similarity tertinggi
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    retrieved_items = []
    for idx in top_indices:
        item = metadata[idx]
        retrieved_items.append({
            "chunk_text": item.get("chunk_text", ""),
            "judul": item.get("judul", ""),
            "url": item.get("url", ""),
            "tanggal": item.get("tanggal", ""),
            "doc_id": item.get("doc_id", ""),
            "similarity": float(similarities[idx])
        })
    
    return retrieved_items
