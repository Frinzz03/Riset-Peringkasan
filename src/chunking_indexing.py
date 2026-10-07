import os
import re
import json
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
import config

# Path file penyimpanan index vektor
INDEX_EMBEDDINGS_FILE = os.path.join(config.CHROMA_PERSIST_DIR, "embeddings.npy")
INDEX_METADATA_FILE = os.path.join(config.CHROMA_PERSIST_DIR, "metadata.json")

def get_ollama_embedding(text: str, model_name: str = config.DEFAULT_EMBED_MODEL) -> list:
    """Mengambil vektor embedding dari Ollama API lokal."""
    url = f"{config.OLLAMA_HOST}/api/embeddings"
    payload = {
        "model": model_name,
        "prompt": text
    }
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get("embedding", [])
    except Exception as e:
        print(f"Error generating embedding via Ollama ({model_name}): {e}")
        return []

def get_ollama_embeddings_batch(texts: list, model_name: str = config.DEFAULT_EMBED_MODEL) -> list:
    """
    Mengambil daftar vektor embedding secara batch menggunakan Ollama API (/api/embed).
    Jauh lebih cepat (hingga 10x) dibanding pemanggilan sekuensial satu per satu.
    Otomatis fallback ke pemanggilan perorangan jika terjadi error.
    """
    if not texts:
        return []
    url = f"{config.OLLAMA_HOST}/api/embed"
    payload = {
        "model": model_name,
        "input": texts
    }
    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        embeddings = data.get("embeddings", [])
        if len(embeddings) == len(texts):
            return embeddings
    except Exception as e:
        # Fallback ke sequential jika endpoint /api/embed bermasalah
        pass

    results = []
    for t in texts:
        results.append(get_ollama_embedding(t, model_name=model_name))
    return results

def split_into_sentences(text: str) -> list:
    """
    Memisahkan teks menjadi daftar kalimat utuh dengan melindungi:
    - Pemisah ribuan angka & format desimal (misal: 45.000, 1.500, 3.5)
    - Format waktu (misal: 09.00, 15.30)
    - Singkatan gelar akademis/medis & singkatan baku (misal: dr., Dr., Sp., Prof., apt., dll.)
    - Penomoran butir daftar (misal: 1., 2., 3.)
    """
    if not text or not text.strip():
        return []
    
    # 1. Bersihkan sisa tag video detik & kode inisial jurnalis jika ada
    text = re.sub(r'\[Gambas:[^\]]*\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\([a-z]{2,5}/[a-z]{2,5}\)$', '', text.strip(), flags=re.IGNORECASE)
    
    # 2. Lindungi titik singkatan umum dan format angka agar tidak terbelah
    text = re.sub(r'\b(dr|Dr|Sp|Prof|apt|Ns|Subsp|Neo|No|no|dsb|dll|hlm|Hal|hal|dkk)\.', r'\1__DOT__', text)
    # Lindungi singkatan apapun yang diikuti koma / titik dua / kurung tutup (e.g. Sp.A, Subsp. Neo.,)
    text = re.sub(r'(\b[A-Za-z]{1,6})\.(\s*[,;:)\]])', r'\1__DOT__\2', text)
    # Lindungi angka dengan titik (45.000, 09.00)
    text = re.sub(r'(\d+)\.(\d+)', r'\1__NUMDOT__\2', text)
    
    # 3. Ekstraksi kalimat yang diakhiri tanda baca (. ! ?) atau akhir teks
    raw_sentences = re.findall(r'[^.!?]+(?:[.!?]+["\'”’]?|$)', text, flags=re.UNICODE)
    
    sentences = []
    for s in raw_sentences:
        s_restored = s.replace('__DOT__', '.').replace('__NUMDOT__', '.').strip()
        if not s_restored:
            continue
            
        # Bersihkan bullet hyphens dan simbol aneh di awal kalimat (misal "- Pola makan" -> "Pola makan")
        s_restored = re.sub(r'^[-•*\s]+', '', s_restored)
        if not s_restored:
            continue
            
        # Penanganan kalimat panjang ekstrem (> 900 karakter tanpa tanda titik)
        if len(s_restored) > 900:
            clause_parts = re.findall(r'[^,;]+(?:[,;]+|$)', s_restored)
            sub_buf = ""
            for cp in clause_parts:
                if len(sub_buf) + len(cp) > 500 and sub_buf:
                    sentences.append(sub_buf.strip())
                    sub_buf = cp
                else:
                    sub_buf += (" " if sub_buf else "") + cp
            if sub_buf.strip():
                sentences.append(sub_buf.strip())
            continue

        # Penanganan klausa gantung yang diawali tanda baca
        if sentences and re.match(r'^[,;:)\]]', s_restored):
            sentences[-1] = sentences[-1] + ' ' + s_restored
            continue

        # Penanganan sambungan kalimat yang diawali huruf kecil (misal kutipan dialog berlanjut)
        if sentences and re.match(r'^[a-z]', s_restored):
            sentences[-1] = sentences[-1] + ' ' + s_restored
            continue

        # Penanganan nomor urut daftar (misal '1.' atau '2.')
        if sentences and re.match(r'^\d+\.?$', sentences[-1]):
            sentences[-1] = sentences[-1] + ' ' + s_restored
        elif sentences and re.search(r'\b\d+\.$', sentences[-1]) and len(sentences[-1].split()[-1]) <= 3:
            m_num = re.search(r'(\s+\d+\.)$', sentences[-1])
            if m_num:
                num_part = m_num.group(1).strip()
                sentences[-1] = sentences[-1][:m_num.start()].strip()
                sentences.append(num_part + ' ' + s_restored)
            else:
                sentences.append(s_restored)
        else:
            sentences.append(s_restored)
            
    return sentences

def create_chunks(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
    max_chunk_size: int = 550,
    min_chunk_size: int = 250
) -> list:
    """
    Sentence-Aware Chunking: Membagi teks panjang menjadi potongan chunk semantik 
    berdasarkan batas kalimat utuh tanpa memotong kata atau kalimat di tengah.
    
    - Setiap chunk terdiri dari satu atau beberapa kalimat utuh yang koheren.
    - Target panjang karakter per chunk berkisar antara min_chunk_size hingga max_chunk_size (~500-600 karakter).
    - Overlap dilakukan dengan menyertakan 1 kalimat terakhir dari chunk sebelumnya (semantic overlap)
      untuk menjaga kesinambungan makna antar-chunk saat proses retrieval.
    """
    if not text or not text.strip():
        return []
    
    effective_max = max(max_chunk_size, chunk_size) if chunk_size else max_chunk_size
    effective_min = min(min_chunk_size, effective_max // 2)
    
    sentences = split_into_sentences(text)
    if not sentences:
        return [text.strip()] if text.strip() else []
        
    # Jika seluruh artikel lebih pendek dari batas maksimal, jadikan 1 chunk utuh
    if len(text.strip()) <= effective_max:
        c = text.strip()
        if c[-1] not in ('.', '!', '?', '"', "'", '”', '’'):
            c += '.'
        return [c]
        
    chunks = []
    current_sentences = []
    current_length = 0
    
    for s in sentences:
        s_len = len(s)
        # Jika penambahan kalimat ini melebihi target ukuran chunk dan sudah memenuhi batas minimum
        if current_length + s_len > effective_max and current_length >= effective_min:
            chunk_text = ' '.join(current_sentences)
            if chunk_text[-1] not in ('.', '!', '?', '"', "'", '”', '’'):
                chunk_text += '.'
            chunks.append(chunk_text)
            # Semantic Overlap: sertakan 1 kalimat terakhir untuk kontinuitas semantik
            if len(current_sentences) > 1:
                overlap_sentence = current_sentences[-1]
                current_sentences = [overlap_sentence, s]
                current_length = len(overlap_sentence) + 1 + s_len
            else:
                current_sentences = [s]
                current_length = s_len
        else:
            current_sentences.append(s)
            current_length += (s_len + 1)
            
    if current_sentences:
        chunk_text = ' '.join(current_sentences)
        if chunk_text[-1] not in ('.', '!', '?', '"', "'", '”', '’'):
            chunk_text += '.'
        chunks.append(chunk_text)
        
    return chunks

def build_vector_database(df: pd.DataFrame, limit: int = None, add_docs: int = None, embed_model: str = config.DEFAULT_EMBED_MODEL, reset_index: bool = False):
    """
    Memotong dokumen, menghasilkan embedding via Ollama, dan menyimpan ke file NumPy + JSON.
    Secara default mendukung Incremental Indexing (skip dokumen yang sudah pernah diindeks).
    - limit: Membatasi total baris awal dari dataset yang dicek (misal limit=1500).
    - add_docs: Membatasi jumlah dokumen BARU yang diproses dalam sesi ini (misal add_docs=1000).
    """
    os.makedirs(config.CHROMA_PERSIST_DIR, exist_ok=True)
    print(f"Memulai pembangunan Vector DB di: {config.CHROMA_PERSIST_DIR}", flush=True)
    
    all_embeddings = []
    all_metadata = []
    existing_doc_ids = set()
    existing_titles = set()
    
    # Cek apakah index sebelumnya sudah ada dan tidak meminta reset
    if not reset_index and os.path.exists(INDEX_EMBEDDINGS_FILE) and os.path.exists(INDEX_METADATA_FILE):
        try:
            print("Memuat index yang sudah ada untuk incremental indexing...", flush=True)
            existing_emb_array = np.load(INDEX_EMBEDDINGS_FILE)
            all_embeddings = list(existing_emb_array)
            with open(INDEX_METADATA_FILE, "r", encoding="utf-8") as f:
                all_metadata = json.load(f)
            
            for item in all_metadata:
                if "doc_id" in item:
                    existing_doc_ids.add(str(item["doc_id"]))
                if "judul" in item and item["judul"]:
                    existing_titles.add(item["judul"].strip().lower())
                    
            print(f"Index eksisting ditemukan: {len(existing_doc_ids)} artikel unik ({len(all_metadata)} chunks).", flush=True)
            print("Fitur SKIP aktif: Artikel yang sudah diindeks akan dilewati secara otomatis.", flush=True)
        except Exception as e:
            print(f"Peringatan: Gagal membaca index lama ({e}). Index akan dibuat baru dari awal.", flush=True)
            all_embeddings = []
            all_metadata = []
            existing_doc_ids = set()
            existing_titles = set()
    elif reset_index:
        print("Mode --reset-index aktif: Menghapus index lama dan membangun ulang dari awal.", flush=True)
        
    target_df = df if limit is None else df.head(limit)
    if add_docs is not None:
        print(f"Target sesi ini: Memproses hingga {add_docs} dokumen BARU (dokumen lama dilewati).", flush=True)
    else:
        print(f"Memproses hingga {len(target_df)} berita dari dataset target...", flush=True)
    
    skipped_count = 0
    new_docs_count = 0
    new_chunks_count = 0
    
    for idx, row in tqdm(target_df.iterrows(), total=len(target_df), desc="Indexing Chunks", mininterval=2.0):
        # Cek apakah target dokumen baru sesi ini sudah tercapai
        if add_docs is not None and new_docs_count >= add_docs:
            print(f"\n[TARGET TERCAPAI] Sebanyak {new_docs_count} dokumen baru telah berhasil diproses.", flush=True)
            break
            
        doc_id = str(idx)
        judul = row['judul_clean']
        judul_key = str(judul).strip().lower() if pd.notna(judul) else ""
        
        # Cek apakah dokumen ini sudah pernah di-chunking sebelumnya
        if doc_id in existing_doc_ids or (judul_key and judul_key in existing_titles):
            skipped_count += 1
            continue
            
        isi = row['isi_clean']
        tanggal = str(row.get('tanggal', ''))
        url = str(row.get('url', ''))
        label = str(row.get('label_kelas', ''))
        
        chunks = create_chunks(isi)
        if not chunks:
            continue
            
        # Format teks yang di-embed (menyertakan judul untuk konteks semantik yang lebih kaya)
        embed_texts = [f"Judul: {judul}\nIsi: {chunk}" for chunk in chunks]
        vectors = get_ollama_embeddings_batch(embed_texts, model_name=embed_model)
        
        doc_chunks_added = 0
        for chunk_idx, (chunk, vec) in enumerate(zip(chunks, vectors)):
            if vec and len(vec) == 768:
                all_embeddings.append(vec)
                all_metadata.append({
                    "doc_id": doc_id,
                    "judul": str(judul) if pd.notna(judul) else "",
                    "tanggal": str(tanggal) if pd.notna(tanggal) else "",
                    "url": str(url) if pd.notna(url) else "",
                    "label_kelas": str(label) if pd.notna(label) else "",
                    "chunk_index": int(chunk_idx),
                    "chunk_text": chunk
                })
                new_chunks_count += 1
                doc_chunks_added += 1
        
        if doc_chunks_added > 0:
            new_docs_count += 1
            existing_doc_ids.add(doc_id)
            if judul_key:
                existing_titles.add(judul_key)
        
        # Simpan progres setiap 50 dokumen baru sebagai checkpoint
        if new_docs_count > 0 and new_docs_count % 50 == 0:
            _save_index(all_embeddings, all_metadata)
            print(f"  [Checkpoint] {new_docs_count} dokumen baru ({new_chunks_count} chunk baru) berhasil disimpan.", flush=True)
    
    # Simpan final jika ada dokumen baru
    if new_docs_count > 0:
        _save_index(all_embeddings, all_metadata)
        print(f"\n[SELESAI] Penambahan data berhasil!", flush=True)
    else:
        print(f"\n[INFO] Tidak ada dokumen baru yang perlu ditambahkan.", flush=True)
        
    print(f"Ringkasan Indexing:")
    print(f" - Dokumen dilewati (sudah ada): {skipped_count} berita")
    print(f" - Dokumen baru diproses      : {new_docs_count} berita")
    print(f" - Chunk baru ditambahkan     : {new_chunks_count} chunk")
    print(f" - Total keseluruhan di index : {len(existing_doc_ids)} berita ({len(all_metadata)} chunk)\n", flush=True)

def _save_index(embeddings_list: list, metadata_list: list):
    """Menyimpan embeddings (NumPy) dan metadata (JSON) ke disk."""
    os.makedirs(config.CHROMA_PERSIST_DIR, exist_ok=True)
    
    # Simpan embeddings sebagai file NumPy (.npy)
    emb_array = np.array(embeddings_list, dtype=np.float32)
    np.save(INDEX_EMBEDDINGS_FILE, emb_array)
    
    # Simpan metadata sebagai JSON
    with open(INDEX_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, ensure_ascii=False)

def load_index():
    """Memuat index vektor dari disk."""
    if not os.path.exists(INDEX_EMBEDDINGS_FILE) or not os.path.exists(INDEX_METADATA_FILE):
        raise FileNotFoundError(
            f"Index vektor tidak ditemukan di {config.CHROMA_PERSIST_DIR}. "
            "Jalankan `python main.py --build-index` terlebih dahulu."
        )
    
    embeddings = np.load(INDEX_EMBEDDINGS_FILE)
    with open(INDEX_METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    print(f"Index vektor dimuat: {len(metadata)} chunk, dimensi embedding: {embeddings.shape[1]}")
    return embeddings, metadata
