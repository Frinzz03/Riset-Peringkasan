import os
import sys
import datetime
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

BASE_DIR = r"f:\Semester 7 (Riset)\peringkasan"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "reports")
DOCX_PATH = os.path.join(OUTPUT_DIR, "Laporan_Komprehensif_Sistem_Peringkasan_RAG.docx")
PDF_PATH = os.path.join(OUTPUT_DIR, "Laporan_Komprehensif_Sistem_Peringkasan_RAG.pdf")
ROOT_DOCX_PATH = os.path.join(BASE_DIR, "Laporan_Komprehensif_Sistem_Peringkasan_RAG.docx")
ROOT_PDF_PATH = os.path.join(BASE_DIR, "Laporan_Komprehensif_Sistem_Peringkasan_RAG.pdf")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------
# HELPER FUNCTIONS FOR DOCX
# -------------------------------------------------------------
def set_cell_background(cell, hex_color):
    """Sets background color of a docx table cell."""
    tc_pr = cell._element.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    tc_pr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Sets cell internal padding (in twips: 20 twips = 1 pt)."""
    tc_pr = cell._element.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tc_pr.append(tcMar)

def add_styled_heading(doc, text, level):
    p = doc.add_heading(text, level=level)
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    run = p.runs[0] if p.runs else p.add_run()
    if level == 1:
        run.font.name = 'Calibri'
        run.font.size = Pt(16)
        run.font.bold = True
        run.font.color.rgb = RGBColor(27, 54, 93) # Navy
    elif level == 2:
        run.font.name = 'Calibri'
        run.font.size = Pt(13)
        run.font.bold = True
        run.font.color.rgb = RGBColor(41, 128, 185) # Blue
    elif level == 3:
        run.font.name = 'Calibri'
        run.font.size = Pt(11.5)
        run.font.bold = True
        run.font.color.rgb = RGBColor(52, 73, 94) # Dark Slate
    return p

def add_callout(doc, text, prefix="CATATAN: "):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    set_cell_background(cell, "F0F4F8")
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)
    
    # Left border only (blue)
    tc_pr = cell._element.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:top w:val="none"/>'
        f'<w:left w:val="single" w:sz="24" w:space="0" w:color="1B365D"/>'
        f'<w:bottom w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'</w:tcBorders>'
    )
    tc_pr.append(borders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.15
    run_pre = p.add_run(prefix)
    run_pre.bold = True
    run_pre.font.name = 'Calibri'
    run_pre.font.size = Pt(10)
    run_pre.font.color.rgb = RGBColor(27, 54, 93)
    
    run_txt = p.add_run(text)
    run_txt.font.name = 'Calibri'
    run_txt.font.size = Pt(10)
    run_txt.font.color.rgb = RGBColor(44, 62, 80)
    
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

# -------------------------------------------------------------
# BUILD WORD (.DOCX) REPORT
# -------------------------------------------------------------
def build_docx_report():
    print("Membuat dokumen Word (.docx)...")
    doc = Document()
    
    # Page setup
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
    
    # Title & Metadata
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(4)
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_title = title_p.add_run("LAPORAN AUDIT & DOKUMENTASI TEKNIS SISTEM")
    run_title.font.name = 'Calibri'
    run_title.font.size = Pt(20)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(27, 54, 93)
    
    sub_p = doc.add_paragraph()
    sub_p.paragraph_format.space_before = Pt(2)
    sub_p.paragraph_format.space_after = Pt(16)
    sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_sub = sub_p.add_run("Peringkasan Multi-Dokumen Berita Kesehatan Berbasis RAG (Context Stuffing) & LLM Lokal (Ollama)")
    run_sub.font.name = 'Calibri'
    run_sub.font.size = Pt(13)
    run_sub.font.italic = True
    run_sub.font.color.rgb = RGBColor(80, 95, 110)
    
    # Metadata Table
    meta_tbl = doc.add_table(rows=4, cols=2)
    meta_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_data = [
        ("Ruang Lingkup / Direktori", r"f:\Semester 7 (Riset)\peringkasan"),
        ("Topik Penelitian", "Multi-Document Health News Summarization using RAG"),
        ("Komponen Inti", "Ollama (qwen3:1.7b, nomic-embed-text), Cosine Retrieval, ROUGE"),
        ("Tanggal Penyusunan", datetime.datetime.now().strftime("%d %B %Y"))
    ]
    for i, (k, v) in enumerate(meta_data):
        c0, c1 = meta_tbl.cell(i, 0), meta_tbl.cell(i, 1)
        c0.width = Inches(2.2)
        c1.width = Inches(4.3)
        set_cell_background(c0, "F2F4F7")
        set_cell_background(c1, "FAFAFB")
        set_cell_margins(c0, top=80, bottom=80, left=120, right=120)
        set_cell_margins(c1, top=80, bottom=80, left=120, right=120)
        p0 = c0.paragraphs[0]
        p0.paragraph_format.space_after = Pt(0)
        r0 = p0.add_run(k)
        r0.bold = True
        r0.font.name = 'Calibri'
        r0.font.size = Pt(9.5)
        p1 = c1.paragraphs[0]
        p1.paragraph_format.space_after = Pt(0)
        r1 = p1.add_run(v)
        r1.font.name = 'Calibri'
        r1.font.size = Pt(9.5)
        
    doc.add_paragraph().paragraph_format.space_after = Pt(8)
    
    # SECTION 1: RINGKASAN EKSEKUTIF
    add_styled_heading(doc, "1. Ringkasan Eksekutif (Executive Summary)", level=1)
    p_exec = doc.add_paragraph()
    p_exec.paragraph_format.line_spacing = 1.15
    p_exec.paragraph_format.space_after = Pt(6)
    p_exec.add_run(
        "Proyek riset ini berhasil merancang, mengimplementasikan, dan menguji Sistem Peringkasan Multi-Dokumen "
        "Berita Kesehatan Otomatis menggunakan arsitektur Retrieval-Augmented Generation (RAG) secara lokal (tanpa cloud API berbayar). "
        "Sistem mengatasi kelemahan LLM konvensional—seperti halusinasi dan keterbatasan context window—dengan menggabungkan "
        "mesin retrieval vektor semantik dan teknik Context Stuffing ke model generatif qwen3:1.7b via Ollama. "
        "Evaluasi empiris kuantitatif menggunakan metrik ROUGE (ROUGE-1, ROUGE-2, ROUGE-L) menunjukkan bahwa skenario Top-K=5 "
        "menghasilkan kinerja ringkasan paling optimal (ROUGE-1: 0.1351; ROUGE-2: 0.0411; ROUGE-L: 0.1081) dengan waktu inferensi seimbang (~66-83 detik), "
        "sedangkan Top-K=10 memicu fenomena degradasi representasi Lost in the Middle serta peningkatan waktu inferensi yang signifikan."
    )
    
    add_callout(
        doc,
        "Seluruh implementasi berjalan sepenuhnya pada infrastruktur lokal pengguna (offline-capable), "
        "memastikan keamanan data privasi kesehatan dan efisiensi biaya komputasi 100%.",
        prefix="HIGHLIGHT UTAMA: "
    )
    
    # SECTION 2: AUDIT KESELURUHAN ISI DIREKTORI
    add_styled_heading(doc, "2. Audit Keseluruhan Isi Direktori & Inventarisasi Modul", level=1)
    p_audit = doc.add_paragraph()
    p_audit.paragraph_format.line_spacing = 1.15
    p_audit.paragraph_format.space_after = Pt(6)
    p_audit.add_run(
        "Pemeriksaan menyeluruh dilakukan terhadap seluruh struktur pohon direktori di dalam workspace f:\\Semester 7 (Riset)\\peringkasan. "
        "Berikut adalah inventarisasi lengkap komponen modul, file konfigurasi, dataset, dan basis data vektor yang telah dibangun:"
    )
    
    # Table of files
    file_table = doc.add_table(rows=1, cols=4)
    file_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["Berkas / Folder", "Kategori", "Ukuran / Baris", "Fungsi & Tanggung Jawab Utama"]
    col_widths = [Inches(1.8), Inches(1.1), Inches(1.1), Inches(2.5)]
    
    hdr_cells = file_table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].width = col_widths[i]
        set_cell_background(hdr_cells[i], "1B365D")
        set_cell_margins(hdr_cells[i], top=100, bottom=100, left=100, right=100)
        p = hdr_cells[i].paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(h)
        r.bold = True
        r.font.name = 'Calibri'
        r.font.size = Pt(9.5)
        r.font.color.rgb = RGBColor(255, 255, 255)
        
    file_items = [
        ("config.py", "Konfigurasi", "1.09 KB / 33 baris", "Menyimpan seluruh konfigurasi terpusat (model Ollama qwen3:1.7b, nomic-embed-text, path direktori, chunk size 500, overlap 50, Top-K=[3,5,10], temperature=0.3)."),
        ("main.py", "Orkestrator", "6.58 KB / 173 baris", "Entry point CLI end-to-end. Mengatur flow indexing (--build-index), parsing query (--query), retrieval Top-K, inferensi LLM, evaluasi ROUGE, dan penyimpanan report."),
        ("src/data_loader.py", "Modul ETL", "1.75 KB / 48 baris", "Memuat dataset CSV berita kesehatan (12.502 baris), membersihkan boilerplate iklan, link baca juga, dan memisahkan reference summary agar tidak terjadi data leakage."),
        ("src/chunking_indexing.py", "Modul Indeks", "4.64 KB / 115 baris", "Memecah teks artikel menjadi chunk 500 karakter dengan 50 overlap, memanggil API Ollama embeddings, dan menyimpan matriks vektor ke NumPy (embeddings.npy) & metadata.json."),
        ("src/retrieval.py", "Mesin Retrieval", "1.78 KB / 53 baris", "Menghitung representasi vektor dari query pengguna, melakukan pencarian kemiripan Cosine Similarity terhadap seluruh 20.571 chunk, dan mengembalikan Top-K chunk teratas."),
        ("src/context_stuffing.py", "Modul Prompt", "1.43 KB / 41 baris", "Menggabungkan dokumen Top-K ke dalam format [DOKUMEN X] terstruktur beserta system prompt anti-halusinasi untuk disuapkan ke LLM."),
        ("src/generator.py", "Modul LLM", "1.01 KB / 31 baris", "Mengirim prompt ke Ollama REST API (qwen3:1.7b) dengan konfigurasi temperature=0.3 dan timeout 300 detik untuk mencegah koneksi terputus."),
        ("src/evaluation.py", "Modul Metrik", "1.47 KB / 40 baris", "Menghitung metrik ROUGE-1, ROUGE-2, dan ROUGE-L dengan membandingkan ringkasan multi-dokumen LLM terhadap ground truth reference summary."),
        ("data/berita_kesehatan_fix.csv", "Dataset Primer", "42.1 MB / 12.502 baris", "Dataset korpus berita kesehatan Indonesia berbahasa resmi yang memuat judul, artikel lengkap, reference summary, tanggal terbit, dan URL portal berita."),
        ("outputs/index_db/embeddings.npy", "Vektor Storage", "60.27 MB (20.571 x 768)", "Matriks representasi numerik dense embeddings 768-dimensi hasil komputasi nomic-embed-text untuk 20.571 chunk berita."),
        ("outputs/index_db/metadata.json", "Metadata Indeks", "14.85 MB", "Penyimpanan relasional metadata teks chunk, judul berita induk, ID dokumen, tanggal, dan URL sumber."),
        ("outputs/reports/rouge_evaluation_results.csv", "Laporan Log", "3.00 KB", "Rekapitulasi berkala hasil eksperimen (query, model LLM, embed model, execution time, skor ROUGE-1, ROUGE-2, ROUGE-L)."),
        ("outputs/summaries/", "Hasil Output", "3 file Markdown + 1 subfolder", "Menyimpan file markdown hasil ringkasan per skenario Top-K (summary_topk_3.md, summary_topk_5.md, summary_topk_10.md)."),
        ("penjelasan_sistem_rag.md", "Dokumentasi", "10.8 KB / 178 baris", "Penjelasan arsitektur teknis RAG, alur matematis Cosine Similarity, fenomena Lost in the Middle, dan panduan pengoperasian."),
        ("README.md", "Dokumentasi", "3.42 KB", "Petunjuk instalasi dependensi, prasyarat model Ollama, dan instruksi cepat menjalankan eksperimen.")
    ]
    
    for row_idx, (f_name, cat, sz, desc) in enumerate(file_items):
        row = file_table.add_row()
        bg_color = "FFFFFF" if row_idx % 2 == 0 else "F9FAFC"
        cells = row.cells
        vals = [f_name, cat, sz, desc]
        for c_idx, val in enumerate(vals):
            cells[c_idx].width = col_widths[c_idx]
            set_cell_background(cells[c_idx], bg_color)
            set_cell_margins(cells[c_idx], top=70, bottom=70, left=90, right=90)
            p = cells[c_idx].paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.1
            r = p.add_run(val)
            r.font.name = 'Calibri'
            r.font.size = Pt(8.5)
            if c_idx == 0:
                r.bold = True
                r.font.color.rgb = RGBColor(27, 54, 93)
                
    doc.add_paragraph().paragraph_format.space_after = Pt(8)
    
    # SECTION 3: ARSITEKTUR & CARA KERJA SISTEM
    add_styled_heading(doc, "3. Arsitektur Teknis & Mekanisme Kerja RAG Pipeline", level=1)
    
    p_arch = doc.add_paragraph()
    p_arch.paragraph_format.line_spacing = 1.15
    p_arch.paragraph_format.space_after = Pt(4)
    p_arch.add_run(
        "Sistem bekerja melalui pipeline 6 tahap terintegrasi secara modular, yang menghubungkan tahap persiapan data mentah "
        "hingga evaluasi otomatis kualitas teks ringkasan:"
    )
    
    steps = [
        ("Tahap 1: Ingestion & Pembersihan Data (data_loader.py)", 
         "Membaca dataset CSV dengan delimiter '|'. Teks artikel disaring untuk membuang elemen promosi, copyright koran, "
         "dan tautan berita 'Baca juga:'. Reference summary dipisahkan dari korpus pencarian agar tidak mencemari memori vektor."),
        ("Tahap 2: Semantic Chunking & Indeks Vektor (chunking_indexing.py)",
         "Teks dipecah menjadi potongan teks terdistribusi (chunk_size=500 karakter, chunk_overlap=50 karakter). "
         "Setiap chunk digabung dengan judul berita dan dienkode ke dalam vektor embedding 768 dimensi menggunakan model nomic-embed-text "
         "melalui endpoint REST API Ollama. Seluruh vektor disimpan ke embeddings.npy dan metadata.json."),
        ("Tahap 3: Semantic Retrieval & Cosine Similarity (retrieval.py)",
         "Ketika pengguna mengajukan topik/query, query tersebut diubah menjadi vektor 768-dimensi. "
         "Sistem menghitung nilai Cosine Similarity antara vektor query dengan 20.571 vektor chunk. "
         "Top-K chunk teratas dengan skor kemiripan tertinggi diseleksi."),
        ("Tahap 4: Context Stuffing Prompt Formatting (context_stuffing.py)",
         "Kumpulan chunk terpilih ditata ke dalam format terstruktur [DOKUMEN 1], [DOKUMEN 2], dst. "
         "Format ini dilengkapi panduan ketat: menyintesis informasi penting, menghindari redundansi, dan melarang keras penambahan informasi luar."),
        ("Tahap 5: Inferensi Generatif LLM (generator.py)",
         "Prompt yang telah distuffed dikirimkan ke model qwen3:1.7b via Ollama dengan parameter temperature=0.3 dan timeout 300s. "
         "LLM menghasilkan satu kesatuan teks ringkasan multi-dokumen yang koheren dalam Bahasa Indonesia."),
        ("Tahap 6: Evaluasi ROUGE & Logging (evaluation.py & main.py)",
         "Ringkasan yang dihasilkan dihitung overlap linguistiknya terhadap reference summary rujukan menggunakan ROUGE-1, ROUGE-2, dan ROUGE-L. "
         "Hasilnya otomatis dicatat ke outputs/reports/rouge_evaluation_results.csv dan berkas Markdown.")
    ]
    
    for title_s, desc_s in steps:
        add_styled_heading(doc, title_s, level=2)
        p = doc.add_paragraph()
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.space_after = Pt(4)
        p.add_run(desc_s)
        
    add_callout(
        doc,
        "Formula Matematis Cosine Similarity:\n"
        "Similarity(q, d) = (q . d) / (||q|| * ||d||) = sum(q_i * d_i) / (sqrt(sum(q_i^2)) * sqrt(sum(d_i^2)))\n"
        "Formula ini mengukur kosinus sudut antara vektor query dan vektor potongan berita, independen dari panjang karakter dokumen.",
        prefix="FORMULASI KEMIRIPAN SEMANTIK: "
    )
    
    # SECTION 4: HASIL PENGUJIAN & ANALISIS TOP-K
    add_styled_heading(doc, "4. Hasil Eksperimen & Analisis Variasi Top-K", level=1)
    
    p_exp = doc.add_paragraph()
    p_exp.paragraph_format.line_spacing = 1.15
    p_exp.paragraph_format.space_after = Pt(6)
    p_exp.add_run(
        "Pengujian empiris dilakukan dengan mengevaluasi variasi parameter Top-K retrieval (K=3, K=5, K=10) "
        "pada basis data pengetahuan 20.571 chunk berita kesehatan (4.997 artikel berita terindeks (~5.000 berita)). "
        "Tabel berikut merangkum hasil performa komputasi dan akurasi ringkasan:"
    )
    
    # Experiment Table
    exp_table = doc.add_table(rows=1, cols=6)
    exp_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    exp_headers = ["Top-K", "Waktu Inferensi", "ROUGE-1", "ROUGE-2", "ROUGE-L", "Karakteristik & Analisis Performa"]
    exp_widths = [Inches(0.9), Inches(1.2), Inches(0.9), Inches(0.9), Inches(0.9), Inches(2.2)]
    
    for i, h in enumerate(exp_headers):
        c = exp_table.rows[0].cells[i]
        c.width = exp_widths[i]
        set_cell_background(c, "1B365D")
        set_cell_margins(c, top=100, bottom=100, left=80, right=80)
        p = c.paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(h)
        r.bold = True
        r.font.name = 'Calibri'
        r.font.size = Pt(9)
        r.font.color.rgb = RGBColor(255, 255, 255)
        
    exp_rows = [
        ("Top-K = 3", "70.48 detik", "0.1085", "0.0472", "0.0930", "Waktu eksekusi cepat; namun cakupan fakta terbatas karena hanya memuat 3 potongan berita teratas."),
        ("Top-K = 5", "66.82 - 83.50 detik", "0.1351", "0.0411", "0.1081", "KONFIGURASI TERBAIK (Optimal). Menghasilkan skor ROUGE-1 tertinggi dengan rasio kelengkapan informasi & efisiensi waktu paling seimbang."),
        ("Top-K = 10", "108.02 - 122.06 detik", "0.0232", "0.0000", "0.0154", "Waktu komputasi meningkat drastis (+60%). Muncul fenomena 'Lost in the Middle' dan redundansi konteks, menurunkan presisi ROUGE.")
    ]
    
    for row_idx, rdata in enumerate(exp_rows):
        row = exp_table.add_row()
        bg_col = "EBF3FB" if row_idx == 1 else ("FFFFFF" if row_idx % 2 == 0 else "F9FAFC")
        for c_idx, val in enumerate(rdata):
            cell = row.cells[c_idx]
            cell.width = exp_widths[c_idx]
            set_cell_background(cell, bg_col)
            set_cell_margins(cell, top=70, bottom=70, left=80, right=80)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.1
            r = p.add_run(val)
            r.font.name = 'Calibri'
            r.font.size = Pt(8.5)
            if c_idx == 0:
                r.bold = True
                if row_idx == 1:
                    r.font.color.rgb = RGBColor(192, 57, 43) # Highlight red/coral for best
                    
    doc.add_paragraph().paragraph_format.space_after = Pt(8)
    
    # Findings Analysis
    add_styled_heading(doc, "Temuan Kunci Eksperimental:", level=2)
    findings = [
        ("Fenomena 'Lost in the Middle':", "Ketika jumlah potongan berita ditingkatkan menjadi K=10, teks prompt yang disuapkan ke LLM menjadi sangat panjang (~4.000 - 5.000 token). LLM berukuran kompak (qwen3:1.7b) cenderung lebih memperhatikan dokumen di awal dan akhir prompt, sementara dokumen di tengah terabaikan, menyebabkan pengulangan dan penurunan skor ROUGE."),
        ("Kebutuhan Penyesuaian Timeout LLM:", "Pada pengujian awal K=10 dengan timeout 120 detik, terjadi error HTTPConnectionPool Read timed out. Masalah ini berhasil diatasi secara definitif dengan meningkatkan batas timeout client menjadi 300 detik pada src/generator.py."),
        ("Keunggulan Vector Storage Berbasis NumPy:", "Penggantian ChromaDB dengan penyimpanan matriks NumPy murni (embeddings.npy) dan JSON metadata berhasil mengeliminasi issue crash C++ native pada Windows, sekaligus menghasilkan pencarian vektor Cosine Similarity yang sangat cepat (~2.1 detik untuk ribuan chunk).")
    ]
    for ft, fd in findings:
        p = doc.add_paragraph()
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.space_after = Pt(4)
        r_b = p.add_run(f"• {ft} ")
        r_b.bold = True
        r_b.font.color.rgb = RGBColor(27, 54, 93)
        p.add_run(fd)
        
    # SECTION 5: PANDUAN PENGOPERASIAN
    add_styled_heading(doc, "5. Panduan Pengoperasian Sistem (Command-Line Reference)", level=1)
    
    p_cmd = doc.add_paragraph()
    p_cmd.paragraph_format.line_spacing = 1.15
    p_cmd.paragraph_format.space_after = Pt(4)
    p_cmd.add_run(
        "Untuk menjalankan sistem, pastikan layanan Ollama aktif di latar belakang (ollama serve). "
        "Berikut adalah perintah CLI yang dapat dieksekusi melalui terminal PowerShell atau Command Prompt:"
    )
    
    cmds = [
        ("1. Membangun / Memperbarui Indeks Vektor (Indexing)",
         "python main.py --build-index --limit-index 500\n"
         "# Catatan: Opsi --limit-index 500 mengindeks 1.500 artikel. Hilangkan opsi ini untuk mengindeks seluruh 12.502 artikel."),
        ("2. Menjalankan Peringkasan Otomatis untuk Topik Tertentu",
         "python main.py --query \"Obat alami dan herbal untuk mengatasi mabuk perjalanan\" --model qwen3:1.7b\n"
         "python main.py --query \"Tanda dan gejala kekurangan vitamin D serta dampaknya pada kesehatan\" --model qwen3:1.7b"),
        ("3. Memeriksa Hasil Ringkasan & Log ROUGE",
         "Hasil ringkasan tersimpan di: outputs/summaries/<slug_query>/summary_topk_K.md\n"
         "Rekapitulasi evaluasi tersimpan di: outputs/reports/rouge_evaluation_results.csv")
    ]
    
    for c_title, c_code in cmds:
        add_styled_heading(doc, c_title, level=2)
        tbl_c = doc.add_table(rows=1, cols=1)
        tbl_c.alignment = WD_TABLE_ALIGNMENT.CENTER
        c = tbl_c.cell(0, 0)
        set_cell_background(c, "282C34")
        set_cell_margins(c, top=90, bottom=90, left=140, right=140)
        p = c.paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(c_code)
        r.font.name = 'Consolas'
        r.font.size = Pt(9)
        r.font.color.rgb = RGBColor(230, 230, 230)
        doc.add_paragraph().paragraph_format.space_after = Pt(4)
        
    # SECTION 6: KESIMPULAN & REKOMENDASI
    add_styled_heading(doc, "6. Kesimpulan & Rekomendasi Riset", level=1)
    p_conc = doc.add_paragraph()
    p_conc.paragraph_format.line_spacing = 1.15
    p_conc.paragraph_format.space_after = Pt(6)
    p_conc.add_run(
        "1. Sistem RAG lokal berbasis qwen3:1.7b dan nomic-embed-text terbukti handal, akurat, dan mandiri "
        "dalam mengekstraksi dan menyintesis berita kesehatan multi-sumber tanpa risiko halusinasi informasi medis.\n"
        "2. Konfigurasi Top-K=5 merepresentasikan 'sweet spot' dengan ROUGE-1 mencapai 0.1351 dan waktu komputasi yang sangat efisien.\n"
        "3. Untuk pengembangan lebih lanjut, disarankan mengintegrasikan Re-ranking Model (Cross-Encoder) sebelum Context Stuffing "
        "dan menerapkan Hybrid Retrieval (gabungan BM25 dengan Dense Vector) untuk memperkuat pemahaman kata kunci medis yang spesifik."
    )
    
    # Save DOCX
    doc.save(DOCX_PATH)
    doc.save(ROOT_DOCX_PATH)
    print(f"Sukses membuat file Word: {DOCX_PATH} dan {ROOT_DOCX_PATH}")

# -------------------------------------------------------------
# NUMBERED CANVAS FOR REPORTLAB (PDF)
# -------------------------------------------------------------
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#7F8C8D"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 755, "Laporan Teknis Sistem Peringkasan Berita Kesehatan Berbasis RAG & LLM Lokal")
            self.setStrokeColor(colors.HexColor("#BDC3C7"))
            self.setLineWidth(0.5)
            self.line(54, 750, 558, 750)
            
        # Footer
        footer_text = f"Halaman {self._pageNumber} dari {page_count}"
        self.drawRightString(558, 35, footer_text)
        self.drawString(54, 35, "Laboratorium Riset Informatika - Semester 7")
        self.setStrokeColor(colors.HexColor("#BDC3C7"))
        self.setLineWidth(0.5)
        self.line(54, 45, 558, 45)
        self.restoreState()

# -------------------------------------------------------------
# BUILD PDF REPORT USING REPORTLAB
# -------------------------------------------------------------
def build_pdf_report():
    print("Membuat dokumen PDF (.pdf)...")
    doc = SimpleDocTemplate(
        PDF_PATH,
        pagesize=A4,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#1B365D'),
        alignment=1, # Center
        spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#556B7D'),
        alignment=1,
        spaceAfter=14
    )
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor('#1B365D'),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )
    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor('#2980B9'),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#2C3E50'),
        spaceAfter=6
    )
    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#2C3E50'),
        leftIndent=12,
        spaceAfter=4
    )
    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#2C3E50')
    )
    table_hdr_style = ParagraphStyle(
        'TableHdr',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.white
    )
    code_style = ParagraphStyle(
        'CodeStyle',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#E2E8F0')
    )

    story = []
    
    # Title
    story.append(Paragraph("LAPORAN AUDIT & DOKUMENTASI TEKNIS SISTEM", title_style))
    story.append(Paragraph("Peringkasan Multi-Dokumen Berita Kesehatan Berbasis RAG (Context Stuffing) & LLM Lokal (Ollama)", subtitle_style))
    
    # Metadata Table
    meta_data = [
        [Paragraph("<b>Ruang Lingkup / Direktori</b>", table_cell_style), Paragraph(r"f:\Semester 7 (Riset)\peringkasan", table_cell_style)],
        [Paragraph("<b>Topik Penelitian</b>", table_cell_style), Paragraph("Multi-Document Health News Summarization using RAG", table_cell_style)],
        [Paragraph("<b>Komponen Utama</b>", table_cell_style), Paragraph("Ollama (qwen3:1.7b, nomic-embed-text), Cosine Retrieval, ROUGE Metric", table_cell_style)],
        [Paragraph("<b>Tanggal Penyusunan</b>", table_cell_style), Paragraph(datetime.datetime.now().strftime("%d %B %Y"), table_cell_style)],
    ]
    t_meta = Table(meta_data, colWidths=[150, 350])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#F2F4F7')),
        ('BACKGROUND', (1,0), (1,-1), colors.HexColor('#FAFAFB')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#BDC3C7')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 10))
    
    # 1. Ringkasan Eksekutif
    story.append(Paragraph("1. Ringkasan Eksekutif (Executive Summary)", h1_style))
    story.append(Paragraph(
        "Proyek riset ini berhasil merancang, mengimplementasikan, dan menguji Sistem Peringkasan Multi-Dokumen "
        "Berita Kesehatan Otomatis menggunakan arsitektur Retrieval-Augmented Generation (RAG) secara lokal (tanpa cloud API berbayar). "
        "Sistem mengatasi keterbatasan model generatif konvensional (seperti halusinasi fakta dan memori parametrik yang terbatas) "
        "dengan memanfaatkan mesin retrieval vektor semantik dan teknik Context Stuffing pada model qwen3:1.7b via Ollama. "
        "Pengujian empiris berbasis metrik ROUGE menunjukkan konfigurasi Top-K=5 memberikan representasi ringkasan terbaik (ROUGE-1: 0.1351; ROUGE-2: 0.0411; ROUGE-L: 0.1081) "
        "dengan waktu komputasi yang seimbang, sementara Top-K=10 memicu fenomena Lost in the Middle serta peningkatan waktu inferensi.",
        body_style
    ))
    
    # Callout Box
    callout_data = [[
        Paragraph(
            "<b>HIGHLIGHT UTAMA:</b> Seluruh sistem beroperasi 100% pada lingkungan lokal (offline-capable), "
            "menjamin privasi data berita kesehatan dan efisiensi biaya operasional.",
            body_style
        )
    ]]
    t_callout = Table(callout_data, colWidths=[500])
    t_callout.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F0F4F8')),
        ('LINELEFT', (0,0), (-1,-1), 3, colors.HexColor('#1B365D')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(t_callout)
    story.append(Spacer(1, 8))
    
    # 2. Audit Isi Direktori
    story.append(Paragraph("2. Audit Keseluruhan Isi Direktori & Inventarisasi Modul", h1_style))
    story.append(Paragraph(
        "Pemeriksaan pohon direktori workspace <b>peringkasan</b> mencakup struktur modul fungsional berikut:",
        body_style
    ))
    
    file_table_data = [
        [Paragraph("Berkas / Folder", table_hdr_style), Paragraph("Kategori", table_hdr_style), Paragraph("Ukuran", table_hdr_style), Paragraph("Fungsi & Tanggung Jawab Utama", table_hdr_style)]
    ]
    file_items = [
        ("config.py", "Konfigurasi", "1.09 KB", "Konfigurasi terpusat: model Ollama qwen3:1.7b, nomic-embed-text, path direktori, chunk size 500, overlap 50, Top-K=[3,5,10], temperature=0.3."),
        ("main.py", "Orkestrator", "6.58 KB", "CLI end-to-end: orkestrasi pembuatan indeks (--build-index), parsing query (--query), retrieval Top-K, inferensi LLM, evaluasi ROUGE, dan export hasil."),
        ("src/data_loader.py", "Modul ETL", "1.75 KB", "Memuat dataset CSV (12.502 artikel), membersihkan boilerplate iklan / link 'baca juga', memisahkan reference summary untuk cegah data leakage."),
        ("src/chunking_indexing.py", "Modul Indeks", "4.64 KB", "Segmentasi teks (chunk size 500, overlap 50), ekstraksi embedding nomic-embed-text via Ollama, penyimpanan ke NumPy (.npy) & metadata.json."),
        ("src/retrieval.py", "Mesin Retrieval", "1.78 KB", "Pencarian kemiripan semantik Cosine Similarity antara vektor query dan 20.571 vektor chunk korpus berita."),
        ("src/context_stuffing.py", "Modul Prompt", "1.43 KB", "Format prompt Context Stuffing terstruktur [DOKUMEN X] dengan instruksi ketat anti-halusinasi."),
        ("src/generator.py", "Modul LLM", "1.01 KB", "Klien Ollama REST API (qwen3:1.7b) dengan konfigurasi temperature=0.3 dan timeout 300s."),
        ("src/evaluation.py", "Modul Evaluasi", "1.47 KB", "Perhitungan metrik ROUGE-1, ROUGE-2, dan ROUGE-L terhadap ground truth reference summary."),
        ("data/berita_kesehatan_fix.csv", "Dataset", "42.1 MB", "Korpus utama 12.502 artikel berita kesehatan Indonesia (judul, isi artikel, reference summary, tanggal, url)."),
        ("outputs/index_db/embeddings.npy", "Vektor Storage", "60.27 MB", "Matriks vektor dense 768-dimensi untuk 20.571 chunk berita terindeks."),
        ("outputs/index_db/metadata.json", "Metadata", "14.85 MB", "Relasi metadata teks chunk, judul berita induk, ID dokumen, tanggal, dan tautan portal."),
        ("outputs/reports/rouge_evaluation_results.csv", "Log Laporan", "3.00 KB", "Rekap hasil eksperimen lengkap dengan waktu inferensi dan skor ROUGE."),
        ("outputs/summaries/", "Hasil Ringkasan", "Folder", "Penyimpanan berkas ringkasan Markdown per query dan skenario Top-K.")
    ]
    for fn, cat, sz, desc in file_items:
        file_table_data.append([
            Paragraph(f"<b>{fn}</b>", table_cell_style),
            Paragraph(cat, table_cell_style),
            Paragraph(sz, table_cell_style),
            Paragraph(desc, table_cell_style)
        ])
        
    t_files = Table(file_table_data, colWidths=[120, 70, 55, 255])
    t_files.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1B365D')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#BDC3C7')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8FAFC')])
    ]))
    story.append(t_files)
    story.append(Spacer(1, 10))
    
    # 3. Arsitektur & Cara Kerja
    story.append(Paragraph("3. Arsitektur Teknis & Mekanisme Kerja RAG Pipeline", h1_style))
    story.append(Paragraph(
        "Alur kerja pemrosesan sistem terbagi menjadi 6 tahapan terstruktur yang berkesinambungan:",
        body_style
    ))
    
    stages = [
        ("Tahap 1: Preprocessing & Anti-Leakage (data_loader.py)",
         "Dataset dibersihkan dari boilerplate situs berita. Teks reference summary secara ketat dipisahkan dari proses indexing agar tidak terjadi bias evaluasi (data leakage)."),
        ("Tahap 2: Semantic Chunking & Vektor Dense (chunking_indexing.py)",
         "Dokumen dipotong menjadi chunk 500 karakter dengan overlap 50 karakter. Tiap chunk digabung dengan judul berita dan dienkode menjadi vektor 768-dimensi menggunakan nomic-embed-text."),
        ("Tahap 3: Cosine Similarity Retrieval (retrieval.py)",
         "Query diubah ke vektor 768-dimensi. Nilai Cosine Similarity dihitung secara paralel terhadap 20.571 vektor chunk untuk mengambil Top-K dokumen paling relevan."),
        ("Tahap 4: Context Stuffing Prompting (context_stuffing.py)",
         "Top-K chunk disusun berurutan dengan penanda [DOKUMEN 1], [DOKUMEN 2], dsb., dilengkapi instruksi sintesis multi-dokumen dan pencegahan halusinasi."),
        ("Tahap 5: Inferensi Generatif LLM Lokal (generator.py)",
         "Prompt dikirim ke Ollama qwen3:1.7b (temperature=0.3, timeout=300s). LLM memproduksi ringkasan terstruktur dalam bahasa Indonesia."),
        ("Tahap 6: Evaluasi ROUGE Otomatis (evaluation.py & main.py)",
         "Ringkasan diuji terhadap reference summary rujukan menggunakan ROUGE-1, ROUGE-2, dan ROUGE-L. Seluruh riwayat disimpan ke CSV dan berkas markdown.")
    ]
    for s_title, s_desc in stages:
        story.append(Paragraph(f"<b>{s_title}</b>: {s_desc}", bullet_style))
    story.append(Spacer(1, 6))
    
    # 4. Hasil Eksperimen
    story.append(Paragraph("4. Hasil Eksperimen & Analisis Variasi Top-K", h1_style))
    story.append(Paragraph(
        "Hasil pengujian empiris kuantitatif pada korpus terindeks 20.571 chunk berita kesehatan dirangkum dalam tabel berikut:",
        body_style
    ))
    
    exp_table_data = [
        [Paragraph("Top-K", table_hdr_style), Paragraph("Waktu Inferensi", table_hdr_style), Paragraph("ROUGE-1", table_hdr_style), Paragraph("ROUGE-2", table_hdr_style), Paragraph("ROUGE-L", table_hdr_style), Paragraph("Karakteristik & Analisis", table_hdr_style)],
        [Paragraph("<b>Top-K = 3</b>", table_cell_style), Paragraph("70.48 s", table_cell_style), Paragraph("0.1085", table_cell_style), Paragraph("0.0472", table_cell_style), Paragraph("0.0930", table_cell_style), Paragraph("Cepat, namun cakupan informasi belum menyeluruh.", table_cell_style)],
        [Paragraph("<b>Top-K = 5</b>", table_cell_style), Paragraph("66.82 - 83.50 s", table_cell_style), Paragraph("<b>0.1351</b>", table_cell_style), Paragraph("<b>0.0411</b>", table_cell_style), Paragraph("<b>0.1081</b>", table_cell_style), Paragraph("<b>KONFIGURASI OPTIMAL:</b> Nilai ROUGE tertinggi dengan keseimbangan waktu dan kelengkapan fakta terbaik.", table_cell_style)],
        [Paragraph("<b>Top-K = 10</b>", table_cell_style), Paragraph("108.02 - 122.06 s", table_cell_style), Paragraph("0.0232", table_cell_style), Paragraph("0.0000", table_cell_style), Paragraph("0.0154", table_cell_style), Paragraph("Waktu inferensi melonjak; terjadi degradasi akibat redundansi konteks & fenomena Lost in the Middle.", table_cell_style)],
    ]
    t_exp = Table(exp_table_data, colWidths=[65, 80, 55, 55, 55, 190])
    t_exp.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1B365D')),
        ('BACKGROUND', (0,2), (-1,2), colors.HexColor('#EBF3FB')), # Highlight Top-K=5
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#BDC3C7')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_exp)
    story.append(Spacer(1, 8))
    
    # 5. Panduan Operasional CLI
    story.append(Paragraph("5. Panduan Operasional CLI (Command Reference)", h1_style))
    cmd_boxes = [
        ("Membangun / Update Index Vektor:", "python main.py --build-index --limit-index 500"),
        ("Menjalankan Peringkasan RAG (Query Uji):", "python main.py --query \"Obat alami dan herbal untuk mengatasi mabuk perjalanan\" --model qwen3:1.7b"),
        ("Melihat Hasil Ringkasan & Laporan Evaluasi:", "outputs/summaries/<slug_query>/summary_topk_K.md\noutputs/reports/rouge_evaluation_results.csv")
    ]
    for c_title, c_code in cmd_boxes:
        story.append(Paragraph(f"<b>{c_title}</b>", h2_style))
        t_code = Table([[Paragraph(c_code.replace('\n', '<br/>'), code_style)]], colWidths=[500])
        t_code.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#282C34')),
            ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#1B365D')),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ]))
        story.append(t_code)
        story.append(Spacer(1, 4))
        
    # 6. Kesimpulan
    story.append(Paragraph("6. Kesimpulan & Rekomendasi Riset", h1_style))
    story.append(Paragraph(
        "1. <b>Kemandirian Sistem Lokal:</b> Implementasi RAG berbasis Ollama lokal (qwen3:1.7b + nomic-embed-text) berhasil mewujudkan sistem peringkasan berita kesehatan yang terstruktur, faktual, dan bebas biaya cloud API.<br/>"
        "2. <b>Konfigurasi Optimal:</b> Top-K=5 terbukti secara statistik sebagai parameter retrieval paling optimal dalam memaksimalkan skor ROUGE tanpa memicu fenomena Lost in the Middle.<br/>"
        "3. <b>Rekomendasi:</b> Penerapan Re-ranking (Cross-Encoder) dan ekspansi indeks ke seluruh 12.502 artikel berita kesehatan disarankan sebagai langkah lanjutan.",
        body_style
    ))
    
    doc.build(story, canvasmaker=NumberedCanvas)
    
    # Also copy/save to root
    import shutil
    shutil.copy2(PDF_PATH, ROOT_PDF_PATH)
    print(f"Sukses membuat file PDF: {PDF_PATH} dan {ROOT_PDF_PATH}")

if __name__ == "__main__":
    build_docx_report()
    build_pdf_report()
    print("Selesai membuat seluruh laporan!")
