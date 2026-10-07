import os
import sys
import json
import time
import re
import requests
import pandas as pd
from tqdm import tqdm

# Import path root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.data_loader import clean_text

# Pastikan konsol Windows mendukung UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def call_groq_api(prompt: str, api_key: str = None, model_name: str = None, max_retries: int = 3) -> str:
    """Memanggil Groq Cloud API untuk Qwen 27B."""
    api_key = api_key or config.GROQ_API_KEY
    model_name = model_name or config.GROQ_MODEL

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Anda adalah kurator dan editor ahli teks medis yang bertugas menyusun Ground Truth Reference Summary "
                    "untuk riset NLP Peringkasan Multi-Dokumen. Tugas Anda adalah merangkum fakta medis secara komprehensif, "
                    "mengalir dalam Bahasa Indonesia formal, mematuhi fakta 100% tanpa halusinasi, dan tanpa kalimat pembuka klise."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.15,
        "top_p": 0.85,
        "max_tokens": 500
    }

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=65)
            if response.status_code == 200:
                data = response.json()
                choices = data.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "").strip()
                return ""
            elif response.status_code == 429:
                print(f"\n[GROQ TPM] Terkena batas kuota per menit (Percobaan {attempt}/{max_retries}). Menunggu jeda 90 detik agar kuota TPM reset sempurna...")
                time.sleep(90)
                print("[GROQ TPM] Jeda 90 detik selesai. Otomatis merangkum kembali...")
                continue
            else:
                time.sleep(3)
        except Exception:
            time.sleep(3)
    return ""

def call_openrouter_api(prompt: str, api_key: str = None, model_name: str = None, max_retries: int = 3) -> str:
    """Fallback ke OpenRouter API dengan reasoning effort none."""
    api_key = api_key or config.OPENROUTER_API_KEY
    model_name = model_name or config.OPENROUTER_MODEL

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/Frinzz03/LabKu",
        "X-Title": "LabKu Health Summarization"
    }
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Anda adalah kurator dan editor ahli teks medis yang bertugas menyusun Ground Truth Reference Summary "
                    "untuk riset NLP Peringkasan Multi-Dokumen. Tugas Anda adalah merangkum fakta medis secara komprehensif, "
                    "mengalir dalam Bahasa Indonesia formal, mematuhi fakta 100% tanpa halusinasi, dan tanpa kalimat pembuka klise."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.15,
        "top_p": 0.85,
        "max_tokens": 500,
        "reasoning": {"effort": "none"}
    }

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=50)
            if response.status_code == 200:
                data = response.json()
                choices = data.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "").strip()
                return ""
            time.sleep(3)
        except Exception:
            time.sleep(3)
    return ""

def build_synthesis_prompt(cluster: dict, articles_data: list) -> str:
    """Menyusun prompt sintesis multi-dokumen berstandar 3 pilar medis."""
    title = cluster["title"]
    query = cluster["query"]

    docs_text = []
    for i, art in enumerate(articles_data, 1):
        art_title = art.get("title", "")
        # Ambil maksimal 400 kata per artikel agar tidak membebani context window
        body_words = art.get("content", "").split()[:400]
        art_body = " ".join(body_words)
        docs_text.append(f"[DOKUMEN {i}] Judul: {art_title}\n{art_body}")

    combined_docs = "\n\n".join(docs_text)

    prompt = f"""Berikut adalah kumpulan {len(articles_data)} artikel berita kesehatan mengenai topik:
TOPIK: "{title}"
PERTANYAAN/QUERY: "{query}"

KORPUS DOKUMEN:
{combined_docs}

INSTRUKSI PENYUSUNAN GROUND TRUTH SINTESIS MULTI-DOKUMEN:
1. Buat SATU ringkasan acuan komprehensif dalam Bahasa Indonesia formal yang menyatukan fakta-fakta medis dari seluruh dokumen di atas.
2. WAJIB mencakup 3 Pilar Medis:
   - Pilar 1: Definisi klinis dan penyebab / faktor risiko utama.
   - Pilar 2: Tanda dan gejala spesifik (termasuk angka atau ambang batas laboratorium jika ada).
   - Pilar 3: Solusi, pencegahan, pantangan, atau penanganan medis/obat.
3. Eliminasi informasi yang duplikat atau berulang antardokumen.
4. Panjang ringkasan ideal: 90 - 120 kata.
5. DILARANG menggunakan kata pembuka klise (misalnya "Artikel ini membahas...", "Berdasarkan dokumen...", "Ringkasan:"). Langsung mulai dengan kalimat substansi medis.
"""
    return prompt

def clean_summary_text(summary: str) -> str:
    """Membersihkan teks ringkasan dari tag atau format formatting yang tidak diinginkan."""
    # Hapus tag <think> jika ada
    summary = re.sub(r'<think>.*?</think>', '', summary, flags=re.DOTALL)
    # Hapus tanda petik ganda di awal dan akhir
    summary = summary.strip().strip('"').strip("'")
    # Hapus prefix markdown bold seperti **Ringkasan:**
    summary = re.sub(r'^\*\*.*?\*\*\s*:?\s*', '', summary)
    return summary.strip()

def run_multidoc_generation(limit: int = None):
    clusters_file = os.path.join(config.BASE_DIR, "data", "clusters_subtopics_mapped.json")
    dataset_file = os.path.join(config.BASE_DIR, "data", "berita_kesehatan_fix.csv")
    output_file = os.path.join(config.BASE_DIR, "data", "multidoc_benchmarks_143.json")

    print("==================================================================")
    print("      EKSEKUSI PEMBUATAN BENCHMARK MULTI-DOKUMEN (143 KLASTER)")
    print("==================================================================")
    
    with open(clusters_file, "r", encoding="utf-8") as f:
        clusters = json.load(f)
        
    print(f"Total klaster terdaftar: {len(clusters)} klaster.")
    if limit:
        clusters = clusters[:limit]
        print(f"Dibatasi untuk diproses: {len(clusters)} klaster.")

    print(f"Membaca dataset korpus: {dataset_file} ...")
    df = pd.read_csv(dataset_file, sep="|")

    # Cek progress yang sudah ada (Auto-Resume Checkpoint)
    completed_benchmarks = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                for item in existing_data:
                    completed_benchmarks[item["id"]] = item
            print(f"Checkpoint ditemukan! Telah selesai sebelumnya: {len(completed_benchmarks)} / {len(clusters)} klaster.")
        except Exception:
            pass

    results = list(completed_benchmarks.values())

    pbar = tqdm(clusters, desc="Sintesis Ground Truth", unit="klaster")
    for c in pbar:
        c_id = c["id"]
        c_title = c["title"]
        c_query = c["query"]
        c_category = c["category"]

        # Lewati jika sudah pernah diproses (Resume)
        if c_id in completed_benchmarks:
            pbar.set_postfix_str(f"Skip {c_id}")
            continue

        pbar.set_postfix_str(f"Proses {c_id[:15]}...")

        # Kumpulkan teks artikel dari dataset
        articles_data = []
        for art_meta in c.get("articles", []):
            orig_idx = art_meta["index"]
            if orig_idx < len(df):
                raw_text = str(df.iloc[orig_idx]["isi_berita"])
                cleaned = clean_text(raw_text)
                articles_data.append({
                    "title": art_meta.get("title", ""),
                    "content": cleaned
                })

        if not articles_data:
            continue

        prompt = build_synthesis_prompt(c, articles_data)

        # Coba panggil Groq terlebih dahulu
        t0 = time.time()
        summary = call_groq_api(prompt)
        provider_used = "groq"

        # Fallback ke OpenRouter jika Groq kosong/limit
        if not summary or len(summary.split()) < 30:
            summary = call_openrouter_api(prompt)
            provider_used = "openrouter"

        latency = time.time() - t0
        summary_clean = clean_summary_text(summary)
        words_cnt = len(summary_clean.split())

        benchmark_entry = {
            "id": c_id,
            "category": c_category,
            "title": c_title,
            "query": c_query,
            "reference": summary_clean,
            "total_source_articles": len(articles_data),
            "word_count": words_cnt,
            "provider": provider_used,
            "latency_sec": round(latency, 2)
        }

        results.append(benchmark_entry)
        completed_benchmarks[c_id] = benchmark_entry

        # Simpan checkpoint langsung ke disk (Incremental Auto-Save)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Jeda antar-request: Groq dibatasi 2 request/menit (~30 detik jeda)
        if provider_used == "groq":
            time.sleep(30)
        else:
            time.sleep(1)

    print("\n" + "="*60)
    print("      PROSES PEMBUATAN BENCHMARK MULTI-DOKUMEN SELESAI!")
    print(f"Total Benchmark Tersimpan : {len(results)} klaster.")
    print(f"Lokasi Berkas Luaran      : {output_file}")
    print("="*60)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Batasi jumlah klaster untuk pengujian (opsional)")
    args = parser.parse_args()

    run_multidoc_generation(limit=args.limit)
