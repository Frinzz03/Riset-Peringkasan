import os
import time
import argparse
import pandas as pd
import re

import config
from src.data_loader import load_dataset
from src.chunking_indexing import build_vector_database
from src.retrieval import retrieve_top_k
from src.context_stuffing import build_context_stuffing_prompt
from src.generator import generate_summary
from src.evaluation import calculate_rouge_scores

def run_experiment(query: str, reference_text: str = None, top_k_list: list = None, model_name: str = config.DEFAULT_LLM_MODEL, embed_model: str = config.DEFAULT_EMBED_MODEL):
    """
    Menjalankan pengujian eksperimen RAG Multi-Dokumen untuk berbagai variasi Top-K.
    """
    if top_k_list is None:
        top_k_list = config.TOP_K_LIST
        
    print(f"\n==================================================================")
    print(f"MENJALANKAN EKSPERIMEN RAG MULTI-DOKUMEN")
    print(f"Query: '{query}'")
    print(f"Model LLM: {model_name} | Embed Model: {embed_model}")
    if reference_text:
        print(f"Teks Acuan (Reference): '{reference_text}'")
    else:
        print("Teks Acuan (Reference): [Tidak diberikan - Evaluasi ROUGE dilewati]")
    print(f"==================================================================\n")
    
    results = []
    
    for k in top_k_list:
        print(f"\n---> [Skenario Top-K = {k}] <---\n")
        
        # 1. Retrieval
        start_time = time.time()
        retrieved_items = retrieve_top_k(query=query, top_k=k, embed_model=embed_model)
        retrieval_time = round(time.time() - start_time, 2)
        print(f"Retrieval selesai ({len(retrieved_items)} chunks ditemukan dalam {retrieval_time}s)")
        
        # 2. Context Stuffing Prompt
        prompt = build_context_stuffing_prompt(query=query, retrieved_chunks=retrieved_items)
        
        # 3. LLM Generation
        start_gen = time.time()
        print(f"Mengirim prompt Context Stuffing ke Ollama ({model_name})...")
        generated_summary = generate_summary(prompt=prompt, model_name=model_name)
        gen_time = round(time.time() - start_gen, 2)
        print(f"Generasi selesai dalam {gen_time} detik.")
        
        # 4. Evaluasi ROUGE (Hanya jika reference_text diberikan)
        if reference_text and reference_text.strip():
            rouge = calculate_rouge_scores(generated_summary=generated_summary, reference_summary=reference_text)
            rouge_r1, rouge_r2, rouge_rl = rouge['rouge1'], rouge['rouge2'], rouge['rougeL']
            rouge_info = f"ROUGE-1: `{rouge_r1}` | ROUGE-2: `{rouge_r2}` | ROUGE-L: `{rouge_rl}`"
            print(f"Skor Evaluasi ROUGE (Top-K={k}): ROUGE-1={rouge_r1}, ROUGE-2={rouge_r2}, ROUGE-L={rouge_rl}")
        else:
            rouge_r1, rouge_r2, rouge_rl = "N/A", "N/A", "N/A"
            rouge_info = "*Tidak dihitung (Teks acuan tidak dimasukkan)*"
            print(f"Evaluasi ROUGE (Top-K={k}): [Dilewati]")
        
        # Buat nama folder khusus berdasarkan slug query agar tidak menimpa query lain
        query_slug = re.sub(r'[^\w\s-]', '', query).strip().lower()
        query_slug = re.sub(r'[-\s]+', '_', query_slug)[:50]
        
        query_summary_dir = os.path.join(config.SUMMARIES_DIR, query_slug)
        os.makedirs(query_summary_dir, exist_ok=True)
        
        # Simpan hasil ringkasan ke file markdown khusus per query & Top-K
        out_summary_file = os.path.join(query_summary_dir, f"summary_topk_{k}.md")
        with open(out_summary_file, "w", encoding="utf-8") as f:
            f.write(f"# Hasil Ringkasan Multi-Dokumen\n\n")
            f.write(f"**Query Pengguna:** `{query}`\n\n")
            f.write(f"**Konfigurasi Retrieval:** Top-K = {k}\n\n")
            f.write(f"**Model LLM:** {model_name} | **Model Embedding:** {embed_model}\n\n")
            f.write(f"**Waktu Generasi:** {gen_time} detik\n\n")
            f.write(f"**Skor Evaluasi ROUGE:** {rouge_info}\n\n")
            f.write(f"---\n\n## 📝 Teks Ringkasan Multi-Dokumen:\n\n{generated_summary}\n\n")
            f.write(f"---\n\n## 📚 Dokumen Sumber yang Di-retrieved (Context Stuffing):\n\n")
            for idx, item in enumerate(retrieved_items, 1):
                sim_pct = round(item.get('similarity', 0.0) * 100, 2)
                f.write(f"### Dokumen {idx}: {item.get('judul')} (Similarity: {sim_pct}%)\n")
                f.write(f"- **URL:** {item.get('url')}\n")
                f.write(f"- **Tanggal:** {item.get('tanggal')}\n")
                f.write(f"- **Potongan Teks (Chunk):** {item.get('chunk_text')}\n\n")
        
        print(f"Hasil ringkasan berhasil disimpan ke: {out_summary_file}")
        
        results.append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "query": query,
            "top_k": k,
            "model_llm": model_name,
            "embed_model": embed_model,
            "gen_time_sec": gen_time,
            "rouge_1": rouge_r1,
            "rouge_2": rouge_r2,
            "rouge_L": rouge_rl,
            "summary_file": out_summary_file
        })
    
    # Export / Append Laporan Evaluasi ke CSV
    df_new = pd.DataFrame(results)
    report_file = os.path.join(config.REPORTS_DIR, "rouge_evaluation_results.csv")
    
    if os.path.exists(report_file):
        df_old = pd.read_csv(report_file)
        df_res = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_res = df_new
        
    df_res.to_csv(report_file, index=False)
    print(f"\n[SELESAI] Tabel Laporan Evaluasi berhasil diperbarui di: {report_file}")
    
    return df_res

def main():
    parser = argparse.ArgumentParser(description="Pipeline RAG Peringkasan Multi-Dokumen Berita Kesehatan dengan Ollama")
    parser.add_argument("--build-index", action="store_true", help="Jalankan proses chunking & pembangunan Vector DB dari CSV")
    parser.add_argument("--limit-index", type=int, default=None, help="Batasi total baris dokumen awal dataset yang dicek (opsional)")
    parser.add_argument("--add-docs", type=int, default=None, help="Batasi jumlah dokumen BARU yang diproses pada sesi ini (misal: 1000)")
    parser.add_argument("--reset-index", action="store_true", help="Hapus index lama dan bangun ulang dari awal (secara default, dokumen lama di-skip)")
    parser.add_argument("--query", type=str, default="Tanda dan gejala kekurangan vitamin D serta dampaknya pada kesehatan", help="Query topik berita yang ingin dirangkum")
    parser.add_argument("--reference", type=str, default=None, help="Teks acuan/ground truth untuk menghitung evaluasi ROUGE (opsional)")
    parser.add_argument("--top-k", type=int, default=None, help="Tentukan jumlah dokumen Top-K spesifik (misal: 3, 5, atau 10). Jika tidak diisi, menguji semua [3, 5, 10]")
    parser.add_argument("--model", type=str, default=config.DEFAULT_LLM_MODEL, help="Nama model LLM Ollama (contoh: qwen3:1.7b atau qwen3:0.6b)")
    parser.add_argument("--embed-model", type=str, default=config.DEFAULT_EMBED_MODEL, help="Nama model embedding Ollama (contoh: nomic-embed-text)")
    
    args = parser.parse_args()
    
    # 1. Pembangunan Vector DB jika argumen --build-index diberikan
    if args.build_index:
        df = load_dataset(config.DATA_PATH, delimiter=config.CSV_DELIMITER)
        build_vector_database(
            df, 
            limit=args.limit_index, 
            add_docs=args.add_docs, 
            embed_model=args.embed_model, 
            reset_index=args.reset_index
        )
        print("Proses indexing selesai. Anda sekarang bisa menjalankan eksperimen RAG.")
        return
    
    # 2. Menentukan Top-K
    top_k_list = [args.top_k] if args.top_k is not None else config.TOP_K_LIST
    
    # 3. Menjalankan Eksperimen Peringkasan
    run_experiment(
        query=args.query,
        reference_text=args.reference,
        top_k_list=top_k_list,
        model_name=args.model,
        embed_model=args.embed_model
    )

if __name__ == "__main__":
    main()
