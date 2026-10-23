import os
import sys
import numpy as np

# Pastikan root workspace terdaftar di sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

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

if __name__ == "__main__":
    import time
    import argparse

    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Pengujian Komponen Retrieval & Cosine Similarity Search")
    parser.add_argument("--query", "-q", type=str, default="Tanda dan gejala kekurangan vitamin D serta dampaknya pada kesehatan", help="Query topik kesehatan")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Jumlah dokumen teratas yang ditarik (default: 5)")
    parser.add_argument("--embed-model", type=str, default=config.DEFAULT_EMBED_MODEL, help="Model embedding Ollama")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("🔍 PENGUJIAN RETRIEVAL SIMILARITY SEARCH (RAG)")
    print(f"Query           : '{args.query}'")
    print(f"Top-K Target    : {args.top_k}")
    print(f"Embedding Model : {args.embed_model}")
    print("=" * 70 + "\n")

    start_time = time.time()
    results = retrieve_top_k(query=args.query, top_k=args.top_k, embed_model=args.embed_model)
    elapsed_time = time.time() - start_time
    elapsed_ms = elapsed_time * 1000

    top_similarity = results[0]["similarity"] * 100 if results else 0.0

    print("--- [LOG PERFORMA RETRIEVAL] ---")
    print(f"* Waktu Retrieval : {elapsed_ms:.2f} ms ({elapsed_time:.4f} detik)")
    print(f"* Total Dokumen   : {len(results)} chunks ditemukan")
    print(f"* Top Similarity  : {top_similarity:.2f}%")
    print("-" * 70 + "\n")

    print(f"📚 Daftar Dokumen Teratas (Top-{args.top_k}):\n")
    for i, item in enumerate(results, 1):
        sim_pct = item["similarity"] * 100
        preview = item.get("chunk_text", "").replace("\n", " ").strip()
        if len(preview) > 160:
            preview = preview[:160] + "..."
        print(f"[{i}] Judul: {item.get('judul', 'Tanpa Judul')}")
        print(f"    - Similarity : {item['similarity']:.4f} ({sim_pct:.2f}%)")
        print(f"    - Doc ID     : {item.get('doc_id', '-')}")
        print(f"    - Tanggal    : {item.get('tanggal', '-')}")
        print(f"    - URL        : {item.get('url', '-')}")
        print(f"    - Cuplikan   : \"{preview}\"")
        print()

