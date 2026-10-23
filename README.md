# Peringkasan Multi-Dokumen Berita Kesehatan Berbasis RAG & Context Stuffing (Ollama)

Repositori ini berisi implementasi sistem **Peringkasan Multi-Dokumen Berita Kesehatan** menggunakan pendekatan **Retrieval-Augmented Generation (RAG)** dengan **Strategi Context Stuffing** dan **Large Language Model (LLM) Lokal via Ollama**.

---

## 📌 Fitur & Komponen Utama

1. **Preprocessing Dataset Berita Kesehatan**: Parsing otomatis dataset `berita_kesehatan_fix.csv` (12.502 berita, delimiter `|`).
2. **Chunking & Vector Indexing**: Pemotongan teks berita menjadi chunk semantik dan pembuatan indeks vektor di **ChromaDB** menggunakan embedding `nomic-embed-text` dari Ollama.
3. **Retrieval Top-K**: Pencarian similarity search berbasis *cosine similarity* untuk mengambil Top-K dokumen teratas.
4. **Context Stuffing Prompting**: Penggabungan beberapa dokumen hasil retrieval ke dalam satu prompt konteks terstruktur beserta penanda sumber berita.
5. **Generasi LLM Lokal (Ollama)**: Inferensi teks ringkasan multi-dokumen menggunakan model lokal `qwen3:1.7b` atau `qwen3:0.6b`.
6. **Evaluasi Otomatis (ROUGE-1, ROUGE-2, ROUGE-L)**: Pengukuran kuantitatif kesamaan ringkasan terhadap *reference summary*.

---

## 🛠️ Langkah-Langkah Persiapan & Penggunaan

### 1. Persiapan Ollama (LLM & Embedding Lokal)

Pastikan aplikasi **Ollama** sudah terinstal dan berjalan secara lokal (`http://localhost:11434`).

Jalankan perintah berikut di terminal/PowerShell untuk mengunduh model yang dibutuhkan:

```bash
# Pull model LLM generatif
ollama pull qwen3:1.7b
# (atau jika menggunakan qwen3:0.6b):
ollama pull qwen3:0.6b

# Pull model embedding
ollama pull nomic-embed-text
```

---

### 2. Instalasi Dependensi Python

Buka terminal pada direktori proyek (`f:\Semester 7 (Riset)\peringkasan`), lalu jalankan:

```bash
pip install -r requirements.txt
```

---

### 3. Pembangunan / Unduh Vector Database (Indexing Dataset)

#### Opsi A: Unduh Pre-built Vector Database & Dataset Bersih (Disarankan)
Karena file indeks vektor `embeddings.npy` berukuran besar (>200 MB) dan dataset korpus bersih `berita_kesehatan_clean.csv` berukuran ~73 MB, file-file tersebut di-hosting di Google Drive agar repositori GitHub tetap sangat ringan dan cepat di-clone:
- 🔗 **Link Download Google Drive:** `[MASUKKAN_LINK_GOOGLE_DRIVE_DISINI]`
- **Cara Penempatan:**  
  Setelah mengunduh file dari tautan di atas, letakkan masing-masing file pada lokasi berikut:
  ```text
  outputs/index_db/embeddings.npy
  data/berita_kesehatan_clean.csv
  ```

#### Opsi B: Pembangunan Index Mandiri (Incremental)
Jika ingin melakukan indexing mandiri dari dataset mentah:
```bash
# Memproses 100 dokumen baru secara inkremental (dokumen lama dilewati otomatis):
python main.py --build-index --add-docs 100
```

---

### 4. Menjalankan Peringkasan RAG & Evaluasi Top-K

Jalankan script utama untuk melakukan pencarian berita, *Context Stuffing*, generasi ringkasan via Ollama, serta evaluasi ROUGE untuk variasi Top-K (K=3, K=5, K=10):

```bash
python main.py --query "Tanda dan gejala kekurangan vitamin D serta dampaknya pada kesehatan" --model qwen3:1.7b
```

---

## 📁 Struktur Direktori & Hasil Luaran

- `outputs/index_db/`: Tempat penyimpanan database vektor ChromaDB lokal.
- `outputs/summaries/`: File hasil ringkasan multi-dokumen dalam format Markdown (`summary_topk_3.md`, `summary_topk_5.md`, `summary_topk_10.md`).
- `outputs/reports/rouge_evaluation_results.csv`: Tabel laporan perbandingan hasil evaluasi ROUGE-1, ROUGE-2, dan ROUGE-L untuk variasi Top-K.

---

## ⚙️ Konfigurasi kustom (`config.py`)

Anda dapat mengubah konfigurasi berikut di file `config.py`:
- `DEFAULT_LLM_MODEL`: Ganti model Ollama default.
- `DEFAULT_EMBED_MODEL`: Ganti model embedding.
- `CHUNK_SIZE` & `CHUNK_OVERLAP`: Ubah panjang potongan chunk.
- `TOP_K_LIST`: Ubah variasi Top-K eksperimen (misal `[3, 5, 10]`).
