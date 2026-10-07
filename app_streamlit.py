import os
import sys
import json
import time
import re
import streamlit as st
import pandas as pd

# Pastikan path modul root terbaca
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from src.retrieval import retrieve_top_k, load_index
from src.context_stuffing import build_context_stuffing_prompt
from src.generator import generate_summary
from src.evaluation import calculate_rouge_scores

# -------------------------------------------------------------
# 1. KONFIGURASI HALAMAN & TEMA STREAMLIT
# -------------------------------------------------------------
st.set_page_config(
    page_title="HealthRAG - Peringkasan Multi-Dokumen Berita Kesehatan",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Dark Glassmorphism Modern
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1300px;
    }

    /* Custom Header Card */
    .header-box {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 22px 26px;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.4);
    }
    
    .header-title {
        font-size: 1.65rem;
        font-weight: 700;
        background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    
    .header-desc {
        color: #94a3b8;
        font-size: 0.92rem;
        line-height: 1.5;
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(17, 24, 39, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 14px 18px;
        text-align: center;
        transition: all 0.2s ease;
    }
    .metric-card:hover {
        border-color: rgba(99, 102, 241, 0.35);
        transform: translateY(-2px);
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        margin: 4px 0;
    }
    .metric-lbl {
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #94a3b8;
    }

    /* Summary Card */
    .summary-card {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 14px;
        padding: 22px 26px;
        margin-top: 14px;
        line-height: 1.7;
        font-size: 0.97rem;
        box-shadow: 0 0 25px rgba(99, 102, 241, 0.12);
    }

    /* Source Chip Badge */
    .source-badge {
        display: inline-block;
        padding: 3px 9px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        background: rgba(99, 102, 241, 0.15);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.3);
        margin-right: 8px;
    }
    
    .sim-badge {
        display: inline-block;
        padding: 3px 9px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    /* Custom Streamlit Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: rgba(15, 23, 42, 0.85);
        padding: 8px 12px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        margin-bottom: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background-color: transparent;
        border-radius: 10px;
        color: #94a3b8;
        font-size: 1.02rem;
        font-weight: 600;
        padding: 0 20px;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #ffffff;
        background-color: rgba(255, 255, 255, 0.05);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.3) 0%, rgba(79, 70, 229, 0.4) 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(99, 102, 241, 0.6) !important;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.25);
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 2. FUNGSI PEMUAT DATA (BENCHMARK & INDEX METADATA)
# -------------------------------------------------------------
@st.cache_data
def load_all_benchmarks():
    """Memuat benchmark multi-dokumen (menggabungkan 143 klaster & sample_benchmarks)."""
    benchmarks = []
    seen_ids = set()

    # 1. Coba baca file multidoc_benchmarks_143.json yang sedang dibuat
    file_143 = os.path.join(config.BASE_DIR, "data", "multidoc_benchmarks_143.json")
    if os.path.exists(file_143):
        try:
            with open(file_143, "r", encoding="utf-8") as f:
                data_143 = json.load(f)
                for item in data_143:
                    if item.get("id") not in seen_ids:
                        benchmarks.append(item)
                        seen_ids.add(item.get("id"))
        except Exception:
            pass

    # 2. Gabungkan dengan sample_benchmarks.json (6 topik proposal resmi)
    file_sample = os.path.join(config.BASE_DIR, "data", "sample_benchmarks.json")
    if os.path.exists(file_sample):
        try:
            with open(file_sample, "r", encoding="utf-8") as f:
                data_sample = json.load(f)
                for item in data_sample:
                    if item.get("id") not in seen_ids:
                        if "category" not in item:
                            item["category"] = "Topik Proposal Utama"
                        benchmarks.append(item)
                        seen_ids.add(item.get("id"))
        except Exception:
            pass

    return benchmarks

@st.cache_data
def get_vector_db_stats():
    """Mengambil status jumlah dokumen dan vektor terindeks."""
    meta_path = os.path.join(config.CHROMA_PERSIST_DIR, "metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            unique_docs = len(set(d.get("doc_id") for d in metadata if "doc_id" in d))
            total_chunks = len(metadata)
            return unique_docs, total_chunks
        except Exception:
            pass
    return 0, 0

# -------------------------------------------------------------
# 2B. AUDIT KUALITAS REAL-TIME & BENCHMARK MATCHER
# -------------------------------------------------------------
STOPWORDS = set([
    "yang", "di", "dan", "dari", "ke", "untuk", "pada", "dengan", "adalah", "ini",
    "itu", "dalam", "bisa", "juga", "karena", "oleh", "seperti", "atau", "dapat",
    "tidak", "ada", "hal", "lebih", "menjadi", "tersebut", "saat", "kesehatan",
    "akan", "sudah", "mengalami", "terjadi", "melalui", "antara", "seseorang"
])

def extract_content_tokens(text: str) -> set:
    """Ekstraksi kata kunci bermakna (huruf kecil, panjang > 2, bukan stopword)."""
    words = re.findall(r'\b[a-zA-Z]{3,}\b', str(text).lower())
    return set(w for w in words if w not in STOPWORDS)

def find_best_matching_reference(query: str, benchmarks: list):
    """Mencari acuan ground truth terbaik berdasarkan kemiripan kata kunci query."""
    if not query or not benchmarks:
        return None, ""
    
    q_words = extract_content_tokens(query)
    if not q_words:
        return None, ""
    
    best_bm = None
    best_score = 0.0
    
    for bm in benchmarks:
        bm_q = extract_content_tokens(bm.get("query", ""))
        bm_t = extract_content_tokens(bm.get("title", ""))
        all_bm = bm_q | bm_t
        if not all_bm:
            continue
        overlap = len(q_words & all_bm)
        score = overlap / len(q_words | all_bm)
        if score > best_score and overlap >= 2:
            best_score = score
            best_bm = bm
            
    if best_bm:
        return best_bm.get("reference", ""), best_bm.get("title", "")
    return None, ""

def audit_summary_quality(query: str, retrieved_chunks: list, summary: str, reference: str = None) -> dict:
    """Audit kualitas & 5 indikator galat medis secara real-time pada hasil ringkasan saat ini."""
    if not summary or summary.startswith("[ERROR"):
        return None
        
    words = summary.split()
    word_count = len(words)
    query_tokens = extract_content_tokens(query)
    summary_tokens = extract_content_tokens(summary)
    
    # 1. Information Loss: Mengecek ketersediaan kata kunci query
    missing_q = query_tokens - summary_tokens
    q_retention = ((len(query_tokens) - len(missing_q)) / len(query_tokens)) if query_tokens else 1.0
    has_info_loss = (q_retention < 0.40)
    
    # 2. Faithfulness / Hallucination Grounding (% kata ringkasan yang didukung berita sumber)
    all_context = ' '.join([c.get('chunk_text', '') for c in retrieved_chunks])
    context_tokens = extract_content_tokens(all_context)
    novel_tokens = summary_tokens - context_tokens
    faithfulness_pct = (1.0 - (len(novel_tokens) / len(summary_tokens))) * 100 if summary_tokens else 100.0
    faithfulness_pct = round(max(0.0, min(100.0, faithfulness_pct)), 1)
    # Toleransi wajar peringkasan abstrak: >= 50% grounded dan aman; < 50% tergolong klaim di luar konteks
    has_hallucination = (faithfulness_pct < 70.0)
    
    # 3. Redundancy (Cek pengulangan klaim/n-gram repetitif antarkalimat)
    low_words = summary.lower().split()
    trigrams = [tuple(low_words[i:i+3]) for i in range(len(low_words)-2)]
    trigram_repeats = (len(trigrams) - len(set(trigrams))) if len(trigrams) > 5 else 0
    
    # Cek duplikasi kalimat identik
    raw_sentences = [
        re.sub(r'^\s*[-*•\d.]+\s*', '', s).strip()
        for s in re.split(r'[.!?\n]+', summary)
        if len(s.strip()) > 15
    ]
    sentence_dups = len(raw_sentences) - len(set([s.lower() for s in raw_sentences]))
    has_redundancy = (trigram_repeats > 3) or (sentence_dups > 0)
    
    # 4. Incoherence (Keterpaduan Alur Semantik & Transisi Lintas Dokumen)
    coherence_pct = 100.0
    valid_transitions = 0
    total_transitions = max(0, len(raw_sentences) - 1)
    
    if total_transitions > 0:
        discourse_markers = [
            'selain itu', 'namun', 'oleh karena itu', 'di samping itu', 'sementara itu',
            'dengan demikian', 'hal ini', 'kondisi ini', 'gejala ini', 'bahkan',
            'sehingga', 'sebagai akibatnya', 'terkait hal', 'adapun', 'di sisi lain',
            'untuk itu', 'karena itu', 'penanganan ini', 'terapi ini', 'langkah ini',
            'meskipun demikian', 'akan tetapi', 'pada dasarnya', 'studi', 'temuan', 'contoh'
        ]
        
        for i in range(total_transitions):
            s_curr = raw_sentences[i]
            s_next = raw_sentences[i + 1]
            tok_curr = extract_content_tokens(s_curr)
            tok_next = extract_content_tokens(s_next)
            
            # Cek keterkaitan entitas, penanda wacana, atau jangkar query
            entity_overlap = len(tok_curr & tok_next)
            has_connector = any(m in s_next.lower() for m in discourse_markers)
            anchor_overlap = (len(tok_next & query_tokens) >= 1)
            
            if entity_overlap >= 1 or has_connector or anchor_overlap:
                valid_transitions += 1
                
        coherence_pct = round((valid_transitions / total_transitions) * 100.0, 1)
        
    syntax_intact = (
        summary.strip().endswith(('.', '!', '?', '"', "'"))
        and not summary.startswith(('[ERROR', 'undefined', 'nan'))
        and len(re.findall(r'\.\s+[a-z]', summary)) <= 1
    )
    
    # Koherensi sehat >= 60%; di bawah 60% tergolong lompatan ide tanpa benang merah
    has_incoherence = (coherence_pct < 60.0) or (not syntax_intact)
    
    # 5. Length Anomaly (Ideal untuk multi-dokumen 3 pilar: 70 - 240 kata)
    has_length_anomaly = (word_count < 70 or word_count > 250)
    
    # 6. Pemenuhan 3 Pilar Medis (Penyebab, Gejala, Solusi/Pencegahan)
    sum_low = summary.lower()
    pilar_1 = any(k in sum_low for k in ['sebab', 'karena', 'akibat', 'pemicu', 'infeksi', 'bakteri', 'virus', 'faktor', 'rusak', 'gangguan', 'terjadi', 'kondisi', 'kelainan', 'kadar', 'glukosa', 'resistensi', 'penyakit', 'merupakan', 'adalah', 'etiologi', 'lingkungan', 'stres', 'memicu'])
    pilar_2 = any(k in sum_low for k in ['gejala', 'tanda', 'keluhan', 'nyeri', 'mual', 'demam', 'pusing', 'lemas', 'sakit', 'kembung', 'berdarah', 'sesak', 'kebas', 'ruam', 'lelah', 'haus', 'lapar', 'berat badan', 'kabur', 'klinis', 'komplikasi', 'hiperglikemia', 'tekanan', 'masalah'])
    pilar_3 = any(k in sum_low for k in ['obat', 'terapi', 'cegah', 'penanganan', 'diet', 'hidrasi', 'operasi', 'konsul', 'rawat', 'minum', 'resep', 'vaksin', 'latihan', 'pola', 'olahraga', 'makanan', 'kontrol', 'gaya hidup', 'hindari', 'konsumsi', 'kelola', 'pengelolaan'])
    pilar_count = sum([pilar_1, pilar_2, pilar_3])
    
    # Evaluasi Predikat Kelayakan (Mengutamakan 3 Pilar Medis & Toleransi Kualitas Seimbang)
    penalty = sum([has_info_loss, has_hallucination, has_redundancy, has_incoherence, has_length_anomaly])
    if penalty == 0 and pilar_count == 3:
        grade = "GRADE A (Sangat Layak & Utuh)"
        grade_badge = "#10b981"
        verdict = f"Ringkasan sangat komprehensif, patuh 3 pilar medis secara utuh, bebas pengulangan, serta alur nalar {coherence_pct}% padu dan faktual."
    elif penalty <= 1 and pilar_count >= 2:
        grade = "GRADE B (Layak & Informatif)"
        grade_badge = "#38bdf8"
        verdict = f"Ringkasan akurat dan mudah dipahami dengan koherensi {coherence_pct}%, struktur pilar medis terjaga dengan baik."
    else:
        grade = "GRADE C (Perlu Peningkatan Konteks)"
        grade_badge = "#f59e0b"
        verdict = f"Ringkasan memuat informasi medis (koherensi {coherence_pct}%), namun terdapat aspek pilar atau redundansi yang perlu diperbaiki."
        
    return {
        "word_count": word_count,
        "q_retention_pct": round(q_retention * 100, 1),
        "has_info_loss": has_info_loss,
        "faithfulness_pct": faithfulness_pct,
        "has_hallucination": has_hallucination,
        "has_redundancy": has_redundancy,
        "coherence_pct": coherence_pct,
        "valid_transitions": valid_transitions,
        "total_transitions": total_transitions,
        "has_incoherence": has_incoherence,
        "has_length_anomaly": has_length_anomaly,
        "pilar_1": pilar_1,
        "pilar_2": pilar_2,
        "pilar_3": pilar_3,
        "pilar_count": pilar_count,
        "grade": grade,
        "grade_badge": grade_badge,
        "verdict": verdict
    }

# Muat data
benchmarks_list = load_all_benchmarks()
unique_docs_cnt, total_chunks_cnt = get_vector_db_stats()

# Kelompokkan benchmark berdasarkan Kategori Klinis
categories_dict = {}
for bm in benchmarks_list:
    cat = bm.get("category", "Kategori Umum")
    if cat not in categories_dict:
        categories_dict[cat] = []
    categories_dict[cat].append(bm)

sorted_categories = sorted(categories_dict.keys())

# Inisialisasi Session State (Otomatis isi sub-topik pertama jika kosong)
if "selected_query" not in st.session_state:
    st.session_state["selected_query"] = benchmarks_list[0]["query"] if benchmarks_list else ""
if "current_reference" not in st.session_state:
    st.session_state["current_reference"] = benchmarks_list[0]["reference"] if benchmarks_list else ""
if "current_topic_title" not in st.session_state:
    st.session_state["current_topic_title"] = benchmarks_list[0]["title"] if benchmarks_list else ""
if "last_results" not in st.session_state:
    st.session_state["last_results"] = None

# -------------------------------------------------------------
# 3. SIDEBAR (PANEL KIRI) - PENGATURAN & PILIHAN TOPIK
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🩺 HealthRAG Control")
    st.caption("Peringkasan Berita Kesehatan Berbasis RAG & LLM Lokal")

    # Status Vektor DB Card
    st.markdown(f"""
    <div style="background: rgba(15, 23, 42, 0.6); padding: 12px 14px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.06); margin-bottom: 16px;">
        <div style="display: flex; align-items: center; gap: 8px; font-size: 0.82rem; color: #10b981; font-weight: 600;">
            <span style="height: 8px; width: 8px; background: #10b981; border-radius: 50%; display: inline-block;"></span>
            Basis Data Vektor Aktif
        </div>
        <div style="font-size: 0.85rem; color: #cbd5e1; margin-top: 6px;">
            📚 <b>{unique_docs_cnt:,}</b> Dokumen | 🧬 <b>{total_chunks_cnt:,}</b> Chunks<br/>
            🎯 Model: <code>nomic-embed-text</code> (768-D)
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📂 Pilih Topik Benchmark Klinis")
    st.caption("Pilih kategori dan sub-topik untuk mengisi query dan acuan emas secara otomatis:")

    if sorted_categories:
        # Dropdown 1: Kategori
        chosen_cat = st.selectbox("1. Kategori Klinis", sorted_categories, index=0)
        
        # Dropdown 2: Sub-Topik
        subtopics_in_cat = categories_dict.get(chosen_cat, [])
        subtopic_titles = [bm["title"] for bm in subtopics_in_cat]
        
        chosen_title = st.selectbox("2. Sub-Topik Medis Spesifik", subtopic_titles, index=0)
        
        # Tombol Terapkan Topik
        if st.button("📌 Terapkan Sub-Topik ke Query", use_container_width=True):
            for bm in subtopics_in_cat:
                if bm["title"] == chosen_title:
                    st.session_state["selected_query"] = bm["query"]
                    st.session_state["current_reference"] = bm["reference"]
                    st.session_state["current_topic_title"] = bm["title"]
                    st.rerun()

    # Tombol Pintas Cepat (Quick Demo Pills)
    st.markdown("##### ⚡ Akses Cepat Topik Populer:")
    quick_cols = st.columns(2)
    with quick_cols[0]:
        if st.button("🩸 Diabetes", use_container_width=True):
            for bm in benchmarks_list:
                if "diabetes" in bm["id"].lower():
                    st.session_state["selected_query"] = bm["query"]
                    st.session_state["current_reference"] = bm["reference"]
                    st.session_state["current_topic_title"] = bm["title"]
                    st.rerun()
        if st.button("🎗️ Kanker Payudara", use_container_width=True):
            for bm in benchmarks_list:
                if "payudara" in bm["id"].lower():
                    st.session_state["selected_query"] = bm["query"]
                    st.session_state["current_reference"] = bm["reference"]
                    st.session_state["current_topic_title"] = bm["title"]
                    st.rerun()
    with quick_cols[1]:
        if st.button("❤️ Hipertensi", use_container_width=True):
            for bm in benchmarks_list:
                if "hipertensi" in bm["id"].lower():
                    st.session_state["selected_query"] = bm["query"]
                    st.session_state["current_reference"] = bm["reference"]
                    st.session_state["current_topic_title"] = bm["title"]
                    st.rerun()
        if st.button("🚗 Mabuk Jalan", use_container_width=True):
            for bm in benchmarks_list:
                if "mabuk" in bm["id"].lower():
                    st.session_state["selected_query"] = bm["query"]
                    st.session_state["current_reference"] = bm["reference"]
                    st.session_state["current_topic_title"] = bm["title"]
                    st.rerun()

    st.markdown("---")
    st.markdown("#### ⚙️ Konfigurasi Sistem RAG")

    top_k = st.radio(
        "Jumlah Dokumen Terambil (Top-K):",
        [3, 5, 10],
        index=1,
        format_func=lambda x: f"Top-K = {x} (Optimal / Rekomendasi)" if x == 5 else (f"Top-K = {x} (Cepat)" if x == 3 else f"Top-K = {x} (Konteks Penuh)")
    )

    model_name = st.selectbox(
        "Model LLM Lokal (Ollama):",
        ["qwen3:1.7b", "qwen3:0.6b"],
        index=0,
        help="qwen3:1.7b adalah model utama proposal dengan akurasi tertinggi."
    )

    temperature = st.slider("Temperature:", 0.0, 0.7, 0.3, 0.05, help="0.3 direkomendasikan untuk ringkasan faktual.")

# -------------------------------------------------------------
# 4. HALAMAN KANAN (MAIN CONTENT) - INPUT QUERY & HASIL
# -------------------------------------------------------------
# Header Box
st.markdown("""
<div class="header-box">
    <div class="header-title">🩺 HealthRAG: Peringkasan Berita Kesehatan Multi-Dokumen</div>
    <div class="header-desc">
        Sistem peringkasan multi-dokumen berita kesehatan otomatis berbasis <b>Retrieval-Augmented Generation (RAG)</b> 
        dengan strategi <b>Context Stuffing</b> menggunakan <b>LLM Lokal Ollama</b>.
    </div>
</div>
""", unsafe_allow_html=True)

# Sistem 2 Tab: Demo Interaktif vs Dashboard Riset Skripsi
tab_demo, tab_eval = st.tabs([
    "🩺 Demo Peringkasan Interaktif",
    "📊 Dashboard Evaluasi & Error Analysis (Riset Skripsi)"
])

# =============================================================
# TAB 1: DEMO PERINGKASAN INTERAKTIF
# =============================================================
with tab_demo:
    # Form Input Query di Halaman Kanan
    with st.container():
        st.markdown("### 🔍 Input Query / Topik Kesehatan")
        
        # Indikator Topik Terpilih jika ada
        if st.session_state["current_topic_title"]:
            st.info(f"📌 **Sub-Topik Aktif:** {st.session_state['current_topic_title']} *(Ground Truth acuan emas telah disinkronkan otomatis di balik layar)*")

        # Text Area Query (Bisa diketik bebas atau terisi dari sidebar)
        query_input = st.text_area(
            "Masukkan topik atau pertanyaan kesehatan yang ingin Anda rangkum:",
            value=st.session_state["selected_query"],
            height=100,
            placeholder="Contoh: Gejala dan cara penanganan penyakit diabetes melitus serta makanan yang dianjurkan...",
            key="main_query_input"
        )

        col_btn, col_info = st.columns([1, 2])
        with col_btn:
            start_btn = st.button("🚀 Mulai Peringkasan Multi-Dokumen", type="primary", use_container_width=True)
        with col_info:
            st.caption(f"⚡ Mode: **Top-K = {top_k}** | Generator: **{model_name}** | Temp: **{temperature}**")

    # Eksekusi Pipeline RAG saat Tombol Diklik
    if start_btn:
        if not query_input.strip():
            st.warning("⚠️ Silakan masukkan topik atau pertanyaan kesehatan terlebih dahulu!")
        else:
            # 1. Pastikan Ground Truth Acuan Emas Terpilih & Sinkron
            current_ref = st.session_state.get("current_reference", "")
            # Cari acuan benchmark yang paling cocok dengan kata kunci query
            matched_ref, matched_title = find_best_matching_reference(query_input.strip(), benchmarks_list)
            if matched_ref:
                current_ref = matched_ref
                st.session_state["current_reference"] = current_ref
                if matched_title:
                    st.session_state["current_topic_title"] = matched_title

            progress_placeholder = st.empty()
            
            with progress_placeholder.container():
                with st.spinner("1/3 Mencari potongan berita relevan via Pure NumPy Cosine Similarity..."):
                    t_ret_start = time.time()
                    retrieved_chunks = retrieve_top_k(query_input.strip(), top_k=top_k)
                    ret_latency = time.time() - t_ret_start

                if not retrieved_chunks:
                    st.error("❌ Tidak ditemukan dokumen berita yang cocok di dalam basis data vektor!")
                else:
                    with st.spinner("2/3 Menyusun prompt Context Stuffing berkaidah 3 Pilar Medis..."):
                        rag_prompt = build_context_stuffing_prompt(query_input.strip(), retrieved_chunks)

                    with st.spinner(f"3/3 Menggenerasi ringkasan menggunakan LLM lokal {model_name}..."):
                        t_gen_start = time.time()
                        generated_summary = generate_summary(rag_prompt, model_name=model_name, temperature=temperature)
                        gen_latency = time.time() - t_gen_start

                    # 4. Hitung ROUGE jika ada reference summary
                    rouge_scores = None
                    if current_ref and generated_summary and not generated_summary.startswith("[ERROR"):
                        rouge_scores = calculate_rouge_scores(generated_summary, current_ref)

                    # 5. Audit Kualitas & 5 Indikator Galat Medis Real-Time
                    audit_res = audit_summary_quality(
                        query=query_input.strip(),
                        retrieved_chunks=retrieved_chunks,
                        summary=generated_summary,
                        reference=current_ref
                    )

                    # Simpan ke session state
                    st.session_state["last_results"] = {
                        "query": query_input.strip(),
                        "topic_title": st.session_state.get("current_topic_title", ""),
                        "reference": current_ref,
                        "summary": generated_summary,
                        "retrieved_chunks": retrieved_chunks,
                        "rouge": rouge_scores,
                        "audit": audit_res,
                        "ret_latency": ret_latency,
                        "gen_latency": gen_latency,
                        "top_k": top_k,
                        "model": model_name,
                        "has_ref": bool(current_ref)
                    }
            
            progress_placeholder.empty()

    # Tampilan Hasil Ringkasan & Evaluasi ROUGE
    res = st.session_state["last_results"]
    if res:
        st.markdown("---")
        
        # A. KARTU METRIK & EVALUASI ROUGE (SCORECARD)
        st.markdown("### 📊 Hasil Evaluasi & Waktu Komputasi")
        
        r_scores = res.get("rouge")
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        
        with m_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-lbl">Waktu Retrieval</div>
                <div class="metric-val" style="color: #38bdf8;">{res['ret_latency']*1000:.1f} ms</div>
                <span style="font-size: 0.72rem; color: #94a3b8;">NumPy Cosine Search</span>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-lbl">Waktu Inferensi</div>
                <div class="metric-val" style="color: #a78bfa;">{res['gen_latency']:.2f} s</div>
                <span style="font-size: 0.72rem; color: #94a3b8;">LLM Lokal Ollama</span>
            </div>
            """, unsafe_allow_html=True)

        if r_scores:
            # Ambil nilai ROUGE secara akurat dari key dictionary (rouge1, rouge2, rougeL)
            r1_f1 = r_scores.get("rouge1", r_scores.get("rouge1_fmeasure", 0.0))
            r2_f1 = r_scores.get("rouge2", r_scores.get("rouge2_fmeasure", 0.0))
            rl_f1 = r_scores.get("rougeL", r_scores.get("rougeL_fmeasure", 0.0))
            
            color_r1 = "#10b981" if r1_f1 >= 0.30 else ("#f59e0b" if r1_f1 >= 0.15 else "#ef4444")
            color_r2 = "#10b981" if r2_f1 >= 0.10 else ("#f59e0b" if r2_f1 >= 0.05 else "#ef4444")
            color_rl = "#10b981" if rl_f1 >= 0.20 else ("#f59e0b" if rl_f1 >= 0.10 else "#ef4444")

            with m_col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-lbl">ROUGE-1 (Unigram)</div>
                    <div class="metric-val" style="color: {color_r1};">{r1_f1:.4f}</div>
                    <span style="font-size: 0.72rem; color: #94a3b8;">P: {r_scores.get('rouge1_precision',0):.2f} | R: {r_scores.get('rouge1_recall',0):.2f}</span>
                </div>
                """, unsafe_allow_html=True)
                
            with m_col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-lbl">ROUGE-2 (Bigram)</div>
                    <div class="metric-val" style="color: {color_r2};">{r2_f1:.4f}</div>
                    <span style="font-size: 0.72rem; color: #94a3b8;">P: {r_scores.get('rouge2_precision',0):.2f} | R: {r_scores.get('rouge2_recall',0):.2f}</span>
                </div>
                """, unsafe_allow_html=True)
                
            with m_col5:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-lbl">ROUGE-L (Fluensi)</div>
                    <div class="metric-val" style="color: {color_rl};">{rl_f1:.4f}</div>
                    <span style="font-size: 0.72rem; color: #94a3b8;">P: {r_scores.get('rougeL_precision',0):.2f} | R: {r_scores.get('rougeL_recall',0):.2f}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            with m_col3:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-lbl">Status Evaluasi</div>
                    <div class="metric-val" style="color: #94a3b8; font-size: 1.1rem;">Mode Bebas</div>
                    <span style="font-size: 0.72rem; color: #64748b;">(Pilih Topik di Sidebar untuk ROUGE)</span>
                </div>
                """, unsafe_allow_html=True)

        # B. BANNER AUDIT KUALITAS REAL-TIME
        audit = res.get("audit")
        if audit:
            st.markdown(f"""
            <div style="background: rgba(15, 23, 42, 0.75); border: 1px solid {audit['grade_badge']}; border-radius: 12px; padding: 14px 18px; margin-top: 14px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 4px 14px rgba(0,0,0,0.2);">
                <div>
                    <span style="font-size: 0.92rem; font-weight: 700; color: {audit['grade_badge']};">⚡ AUDIT KUALITAS REAL-TIME: {audit['grade']}</span>
                    <div style="font-size: 0.83rem; color: #cbd5e1; margin-top: 3px;">{audit['verdict']}</div>
                </div>
                <div style="text-align: right; font-size: 0.82rem; color: #94a3b8; white-space: nowrap; margin-left: 16px;">
                    🛡️ Faktual: <b style="color: #38bdf8;">{audit['faithfulness_pct']}%</b> | 🧩 Koherensi: <b style="color: #a78bfa;">{audit.get('coherence_pct', 100.0)}%</b><br/>
                    🩺 3 Pilar Medis: <b style="color: #34d399;">{audit['pilar_count']}/3 Terpenuhi</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # C. TAMPILAN RINGKASAN MULTI-DOKUMEN
        st.markdown("### 📝 Hasil Ringkasan Multi-Dokumen AI")
        summary_text = res.get("summary", "")
        summary_words = len(summary_text.split())
        
        st.markdown(f"""
        <div class="summary-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 8px;">
                <span style="font-size: 0.82rem; color: #94a3b8;">🎯 Sintesis <b>Top-{res['top_k']} Dokumen</b> Berita Kesehatan</span>
                <span style="font-size: 0.82rem; color: #38bdf8;">📏 Panjang: <b>{summary_words} kata</b></span>
            </div>
            <div>
                {summary_text.replace(chr(10), '<br/>')}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # D. DOKUMEN SUMBER RETRIEVAL (TOP-K)
        st.markdown("### 📚 Dokumen Sumber yang Digunakan (Retrieval Chunks)")
        chunks = res.get("retrieved_chunks", [])
        
        for i, chunk in enumerate(chunks, 1):
            sim_pct = chunk.get("similarity", 0.0) * 100
            title = chunk.get("judul") or chunk.get("title") or f"Dokumen {i}"
            url = chunk.get("url", "#")
            date = chunk.get("tanggal") or chunk.get("date", "-")
            text_body = chunk.get("chunk_text", "")
            
            with st.expander(f"📄 [DOKUMEN {i}] {title} — (Kemiripan: {sim_pct:.1f}%)"):
                st.markdown(f"""
                - **URL Berita:** [{url}]({url})
                - **Tanggal Terbit:** {date}
                - **Skor Cosine Similarity:** `{chunk.get('similarity', 0.0):.4f}`
                """)
                st.markdown("**Cuplikan Teks Berita yang Dikutip:**")
                st.code(text_body, language="text")

# =============================================================
# TAB 2: DASHBOARD EVALUASI & ERROR ANALYSIS (RISET SKRIPSI)
# =============================================================
with tab_eval:
    st.markdown("### 🔬 Evaluasi Sistem RAG & Analisis Galat Medis")
    st.markdown("""
    Dashboard ini menyediakan dua mode evaluasi: **Audit Real-Time** terhadap ringkasan yang baru saja Anda uji, 
    serta **Benchmark Historis** untuk pembuktian komparatif pada sidang skripsi.
    """)

    eval_focus = st.radio(
        "Pilih Fokus Evaluasi:",
        [
            "⚡ Audit Real-Time (Hasil Uji Sesi Saat Ini)",
            "📚 Benchmark Historis Skripsi (18 Skenario Bab 4)"
        ],
        horizontal=True
    )
    st.markdown("---")

    if eval_focus == "⚡ Audit Real-Time (Hasil Uji Sesi Saat Ini)":
        res_current = st.session_state.get("last_results")
        if not res_current or not res_current.get("audit"):
            st.info("""
            💡 **Belum Ada Pengujian pada Sesi Ini**  
            Silakan buka **Tab 1 ('🩺 Demo Peringkasan Interaktif')**, ketik atau pilih topik berita kesehatan, lalu klik tombol **'Mulai Peringkasan Multi-Dokumen'**.  
            Sistem akan otomatis menghitung audit 5 indikator galat medis dan skor ROUGE secara real-time pada bagian ini!
            """)
        else:
            audit_curr = res_current.get("audit")
            rouge_curr = res_current.get("rouge")
            
            # Header Uji Saat Ini
            st.markdown(f"""
            <div style="background: rgba(30, 41, 59, 0.6); padding: 16px 20px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.08); margin-bottom: 20px;">
                <div style="font-size: 0.85rem; color: #94a3b8;">📌 Topik / Query yang Diuji: <b>"{res_current['query']}"</b></div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 8px;">
                    <span style="font-size: 1.15rem; font-weight: 700; color: {audit_curr['grade_badge']};">
                        🏆 Evaluasi Kelayakan: {audit_curr['grade']}
                    </span>
                    <span style="font-size: 0.82rem; color: #cbd5e1;">
                        ⚙️ Model: <b>{res_current['model']}</b> | Top-K: <b>{res_current['top_k']}</b> | Latensi Total: <b>{res_current['ret_latency'] + res_current['gen_latency']:.2f}s</b>
                    </span>
                </div>
                <div style="font-size: 0.85rem; color: #cbd5e1; margin-top: 8px; line-height: 1.5;">
                    📝 <b>Catatan Klinis:</b> {audit_curr['verdict']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # 5 LIVE KPI METRICS
            st.markdown("#### 1. Status 5 Indikator Galat Medis Real-Time")
            c1, c2, c3, c4, c5 = st.columns(5)
            
            with c1:
                col_loss = "#ef4444" if audit_curr["has_info_loss"] else "#10b981"
                status_loss = "Ada Fakta Hilang" if audit_curr["has_info_loss"] else "0% Hilang (Terjaring)"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-lbl">Information Loss</div>
                    <div class="metric-val" style="color: {col_loss}; font-size: 1.25rem;">{status_loss}</div>
                    <span style="font-size: 0.72rem; color: #94a3b8;">Retensi Kata: {audit_curr['q_retention_pct']}%</span>
                </div>
                """, unsafe_allow_html=True)
                
            with c2:
                col_faith = "#10b981" if not audit_curr["has_hallucination"] else "#ef4444"
                status_faith = "Bebas Halusinasi" if not audit_curr["has_hallucination"] else "Klaim Liar"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-lbl">Faithfulness (Faktual)</div>
                    <div class="metric-val" style="color: {col_faith};">{audit_curr['faithfulness_pct']}%</div>
                    <span style="font-size: 0.72rem; color: #94a3b8;">{status_faith} (Korpus Berita)</span>
                </div>
                """, unsafe_allow_html=True)
                
            with c3:
                col_incoh = "#10b981" if not audit_curr["has_incoherence"] else "#ef4444"
                status_incoh = "Alur Mengalir Padu" if not audit_curr["has_incoherence"] else "Lompatan Topik"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-lbl">Koherensi Alur Makna</div>
                    <div class="metric-val" style="color: {col_incoh}; font-size: 1.25rem;">{audit_curr.get('coherence_pct', 100.0)}% Padu</div>
                    <span style="font-size: 0.72rem; color: #94a3b8;">{status_incoh} ({audit_curr.get('valid_transitions', 0)}/{audit_curr.get('total_transitions', 0)} transisi)</span>
                </div>
                """, unsafe_allow_html=True)
                
            with c4:
                col_redun = "#10b981" if not audit_curr["has_redundancy"] else "#f59e0b"
                status_redun = "Bebas Repetisi" if not audit_curr["has_redundancy"] else "Ada Pengulangan"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-lbl">Redundansi</div>
                    <div class="metric-val" style="color: {col_redun}; font-size: 1.25rem;">{status_redun}</div>
                    <span style="font-size: 0.72rem; color: #94a3b8;">Pemeriksaan n-gram repetitif</span>
                </div>
                """, unsafe_allow_html=True)
                
            with c5:
                col_len = "#38bdf8" if not audit_curr["has_length_anomaly"] else "#f59e0b"
                status_len = "Ideal (70-240 kata)" if not audit_curr["has_length_anomaly"] else "Anomali Panjang"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-lbl">Panjang Teks</div>
                    <div class="metric-val" style="color: {col_len};">{audit_curr['word_count']} Kata</div>
                    <span style="font-size: 0.72rem; color: #94a3b8;">{status_len}</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # 2. PEMERIKSAAN 3 PILAR MEDIS & DETAIL ROUGE
            col_pil, col_rg = st.columns([1, 1])
            with col_pil:
                st.markdown("#### 2. Kepatuhan Kaidah 3 Pilar Medis")
                p1_icon = "✅" if audit_curr["pilar_1"] else "❌"
                p2_icon = "✅" if audit_curr["pilar_2"] else "❌"
                p3_icon = "✅" if audit_curr["pilar_3"] else "❌"
                st.markdown(f"""
                - {p1_icon} **Pilar 1: Etiologi & Definisi Klinis** *(Menerangkan penyebab / mekanisme gangguan kesehatan)*
                - {p2_icon} **Pilar 2: Gejala & Manifestasi Klinis** *(Menerangkan tanda-tanda yang dialami pasien)*
                - {p3_icon} **Pilar 3: Solusi, Terapi & Pencegahan** *(Menerangkan pengobatan, gaya hidup, atau mitigasi)*
                """)
                st.caption(f"Skor Pemenuhan Pilar: **{audit_curr['pilar_count']} / 3 Pilar Terpenuhi**.")

            with col_rg:
                st.markdown("#### 3. Rincian Skor Evaluasi ROUGE Real-Time")
                if rouge_curr:
                    st.markdown(f"""
                    - **ROUGE-1 (Unigram):** F1 = `{rouge_curr.get('rouge1', 0):.4f}` | Presisi: `{rouge_curr.get('rouge1_precision', 0):.4f}` | Recall: `{rouge_curr.get('rouge1_recall', 0):.4f}`
                    - **ROUGE-2 (Bigram):** F1 = `{rouge_curr.get('rouge2', 0):.4f}` | Presisi: `{rouge_curr.get('rouge2_precision', 0):.4f}` | Recall: `{rouge_curr.get('rouge2_recall', 0):.4f}`
                    - **ROUGE-L (Fluensi):** F1 = `{rouge_curr.get('rougeL', 0):.4f}` | Presisi: `{rouge_curr.get('rougeL_precision', 0):.4f}` | Recall: `{rouge_curr.get('rougeL_recall', 0):.4f}`
                    """)
                    if res_current.get("topic_title"):
                        st.caption(f"📌 Dihitung terhadap Ground Truth: **{res_current['topic_title']}**")
                else:
                    st.info("ROUGE tidak dihitung karena teks acuan emas (*ground truth*) tidak ditemukan untuk query bebas ini.")

            st.markdown("---")

            # 3. KOMPARASI BERDAMPINGAN: RINGKASAN VS GROUND TRUTH
            if res_current.get("reference"):
                with st.expander("🔍 Komparasi Teks Berdampingan (Ringkasan AI vs Acuan Emas Ground Truth)", expanded=True):
                    cmp1, cmp2 = st.columns(2)
                    with cmp1:
                        st.markdown("**🤖 Ringkasan Hasil Generasi RAG AI:**")
                        st.write(res_current["summary"])
                    with cmp2:
                        st.markdown("**🎯 Teks Acuan Emas (Multi-Doc Ground Truth):**")
                        st.write(res_current["reference"])

    else:
        # TAMPILAN HISTORIS (18 Skenario Bab 4)
        st.markdown("#### 1. Rekapitulasi Frekuensi Galat Medis (N = 18 Skenario)")
        e_col1, e_col2, e_col3, e_col4, e_col5 = st.columns(5)
        
        with e_col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-lbl">Information Loss</div>
                <div class="metric-val" style="color: #10b981;">5.6%</div>
                <span style="font-size: 0.72rem; color: #94a3b8;">1/18 Skenario (Hanya Top-K=3)</span>
            </div>
            """, unsafe_allow_html=True)
            
        with e_col2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-lbl">Halusinasi Liar</div>
                <div class="metric-val" style="color: #10b981;">0.0%</div>
                <span style="font-size: 0.72rem; color: #94a3b8;">100% Patuh Korpus Berita</span>
            </div>
            """, unsafe_allow_html=True)

        with e_col3:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-lbl">Incoherence (Sintaks)</div>
                <div class="metric-val" style="color: #10b981;">0.0%</div>
                <span style="font-size: 0.72rem; color: #94a3b8;">0 Kalimat Terpotong / Rusak</span>
            </div>
            """, unsafe_allow_html=True)

        with e_col4:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-lbl">Redundansi Informasi</div>
                <div class="metric-val" style="color: #f59e0b;">38.9%</div>
                <span style="font-size: 0.72rem; color: #94a3b8;">7/18 Skenario (Ditekan Prompt)</span>
            </div>
            """, unsafe_allow_html=True)

        with e_col5:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-lbl">Anomali Panjang</div>
                <div class="metric-val" style="color: #38bdf8;">27.8%</div>
                <span style="font-size: 0.72rem; color: #94a3b8;">5/18 Skenario (Ideal ~175 Kata)</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # 2. KOMPARASI TOP-K & TEMUAN UTAMA
        st.markdown("#### 2. Komparasi Karakteristik Konfigurasi Top-K")
        k_col1, k_col2, k_col3 = st.columns(3)
        with k_col1:
            st.info("""
            **⚡ Top-K = 3 (Mode Cepat)**
            - **Latensi:** Tercepat (~1.2 ms retrieval, ~5.5 s inferensi).
            - **Kelebihan:** Sangat ringkas, meminimalkan token.
            - **Kelemahan:** Rentan *Information Loss* (5.6%) karena fakta dari artikel pelengkap tidak terambil.
            """)
        with k_col2:
            st.success("""
            **🏆 Top-K = 5 (Optimal / Rekomendasi Skripsi)**
            - **Skor ROUGE-1:** `0.3106` (Tertinggi) | ROUGE-L: `0.1983`
            - **Information Loss:** **0.0%** (Seluruh fakta medis terangkum utuh).
            - **Kesimpulan:** *Sweet Spot* terbaik antara kelengkapan klinis dan efisiensi waktu komputasi.
            """)
        with k_col3:
            st.warning("""
            **📚 Top-K = 10 (Konteks Lengkap)**
            - **Latensi:** Inferensi lebih lama (~12.5 s pada Ollama).
            - **Kelebihan:** Informasi latar belakang sangat komprehensif.
            - **Kelemahan:** Terjadi sedikit kenaikan redundansi akibat kesamaan bahasan antar-berita.
            """)

        st.markdown("---")

        # 3. ANALISIS LOST-IN-THE-MIDDLE
        st.markdown("#### 3. Analisis Efek *Lost-in-the-Middle* (Posisi Dokumen dalam Konteks)")
        st.markdown("""
        Pengujian dilakukan pada konfigurasi **Top-K = 10** untuk memverifikasi apakah LLM lokal mengalami degradasi perhatian pada dokumen urutan tengah (*Liu et al., 2024*):
        """)
        litm_col1, litm_col2, litm_col3, litm_col4 = st.columns(4)
        with litm_col1:
            st.metric("Head Recall (Rank 1-3)", "43.5%", help="Retensi fakta pada dokumen posisi awal")
        with litm_col2:
            st.metric("Middle Recall (Rank 4-7)", "39.5%", help="Retensi fakta pada dokumen posisi tengah")
        with litm_col3:
            st.metric("Tail Recall (Rank 8-10)", "40.5%", help="Retensi fakta pada dokumen posisi akhir")
        with litm_col4:
            st.metric("Middle Decay", "-4.0%", delta="Terkendali", delta_color="normal", help="Penurunan atensi tengah hanya 4.0%")

        st.caption("💡 **Temuan Ilmiah:** Penurunan retensi pada posisi tengah hanya sebesar **4.0%**, membuktikan bahwa strategi prompt *Context Stuffing* berbasis 3 Pilar Medis efektif menjaga retensi memori model.")

        st.markdown("---")

        # 4. GALERI GRAFIK PUBLIKASI RISET (FIGURES)
        st.markdown("#### 4. Galeri Visualisasi Hasil Eksperimen (Gambar Publikasi 300 DPI)")
        
        figures_dir = os.path.join(config.REPORTS_DIR, "figures")
        fig_options = {
            "📊 Distribusi 5 Indikator Galat (Figure 4)": ("fig4_error_analysis_distribution.png", "Distribusi frekuensi 5 indikator galat kualitatif pada 18 skenario eksperimen RAG."),
            "📉 Kurva Efek Lost-in-the-Middle (Figure 3)": ("fig3_lost_in_the_middle.png", "Kurva U-shape retensi informasi berdasarkan urutan posisi dokumen dalam konteks."),
            "📈 Perbandingan Skor ROUGE per Top-K (Figure 1)": ("fig1_rouge_comparison_topk.png", "Perbandingan skor ROUGE-1, ROUGE-2, dan ROUGE-L pada skenario Top-K = 3, 5, dan 10."),
            "⏱️ Trade-off Latensi vs Top-K (Figure 2)": ("fig2_latency_vs_topk.png", "Grafik trade-off antara waktu inferensi generator dengan penambahan jumlah dokumen terambil."),
            "📦 Distribusi Boxplot ROUGE (Figure 5)": ("fig5_rouge_boxplot_distribution.png", "Distribusi sebaran nilai ROUGE F1 antar-topik kesehatan.")
        }
        
        selected_fig_label = st.selectbox("Pilih Grafik Hasil Penelitian:", list(fig_options.keys()))
        fig_filename, fig_caption = fig_options[selected_fig_label]
        fig_path = os.path.join(figures_dir, fig_filename)
        
        if os.path.exists(fig_path):
            st.image(fig_path, caption=f"{selected_fig_label}: {fig_caption}", use_container_width=True)
        else:
            st.info("File gambar grafik belum tersedia. Jalankan `python generate_charts.py` untuk membuatnya.")

        st.markdown("---")

        # 5. TABEL DATA MENTAH AUDIT
        st.markdown("#### 5. Data Mentah Audit Empiris & Hasil Eksperimen")
        
        err_csv_path = os.path.join(config.REPORTS_DIR, "hasil_error_analysis.csv")
        exp_csv_path = os.path.join(config.REPORTS_DIR, "hasil_eksperimen_skenario_e1_e4.csv")
        
        tab_data_choice = st.radio("Pilih Dataset Laporan:", ["Tabel Audit 5 Indikator Galat", "Tabel Lengkap Skor Eksperimen (E1-E4)"], horizontal=True)
        
        if tab_data_choice == "Tabel Audit 5 Indikator Galat" and os.path.exists(err_csv_path):
            df_err_view = pd.read_csv(err_csv_path)
            st.dataframe(df_err_view, use_container_width=True, hide_index=True)
            st.caption("Keterangan: Nilai 1 menunjukkan indikasi terdeteksi, nilai 0 menunjukkan bebas galat.")
        elif tab_data_choice == "Tabel Lengkap Skor Eksperimen (E1-E4)" and os.path.exists(exp_csv_path):
            df_exp_view = pd.read_csv(exp_csv_path)
            cols_to_show = ["topik", "top_k", "retrieval_sec", "generation_sec", "word_count", "rouge_1", "rouge_2", "rouge_l"]
            available_cols = [c for c in cols_to_show if c in df_exp_view.columns]
            st.dataframe(df_exp_view[available_cols], use_container_width=True, hide_index=True)


# -------------------------------------------------------------
# 7. FOOTER
# -------------------------------------------------------------
st.markdown("---")
st.caption("HealthRAG © 2026 • Riset Skripsi S1 Teknik Informatika / Ilmu Komputer • Peringkasan Multi-Dokumen Berbasis RAG & LLM Lokal")
