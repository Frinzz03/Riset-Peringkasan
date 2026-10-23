import os
import re
import sys
import time
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Konfigurasi Path File
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "berita_kesehatan_fix.csv")
CLEAN_DATA_PATH = os.path.join(BASE_DIR, "data", "berita_kesehatan_clean.csv")
CSV_DELIMITER = "|"

def clean_news_text(text: str) -> str:
    """
    Membersihkan isi teks berita kesehatan dari seluruh elemen non-berita:
    1. Normalisasi karakter tak terlihat & unicode bermasalah.
    2. Tag video Detik ([Gambas:...]) dan embed video.
    3. Inisial jurnalis / kode redaksi di akhir atau dalam kurung (e.g. (akd/akd), (sao/dpy)).
    4. Header tanggal & jam publikasi portal di awal teks.
    5. Dateline nama kota dan portal berita (e.g. 'JAKARTA, KOMPAS.com - ', 'Jakarta - detikHealth').
    6. Sisipan tautan rekomendasi ('Baca juga:', 'Simak juga:', 'Baca selengkapnya').
    7. Footer promosi, ajakan langganan, hak cipta, saluran WhatsApp, Google News, dan aplikasi.
    8. Baris metadata redaksi (Penulis:, Editor:, Reporter:, Sumber:).
    9. Tanda baca menggantung dan normalisasi spasi.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # 0. Normalisasi karakter spasi tak terlihat & unicode non-standar
    text = text.replace('\xa0', ' ').replace('\u200b', ' ').replace('\ufeff', ' ').replace('\ufffd', ' ')
    
    # 1. Hapus tag video Detik [Gambas:...] & embed video 'Simak Video "..."'
    text = re.sub(r'\[gambas:[^\]]*\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Simak\s+Video\s+["\'\u201c\u201d][^"\'\u201c\u201d]*["\'\u201c\u201d]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(?:Simak|Tonton)\s+video\s+(?:pilihan\s+berikut|lainnya)[\s:]*', '', text, flags=re.IGNORECASE)
    
    # 2. Hapus header tanggal/jam di awal teks (Detik)
    text = re.sub(r'^[^\n]{0,100}?(?:Senin|Selasa|Rabu|Kamis|Jumat|Sabtu|Minggu),\s*\d{1,2}\s+[A-Za-z]+\s+\d{4}\s+\d{1,2}:\d{2}\s*WIB\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(?:Foto\s+(?:Health|News|Detik)[^\n]*)?(?:Senin|Selasa|Rabu|Kamis|Jumat|Sabtu|Minggu),\s*\d{1,2}\s+[A-Za-z]+\s+\d{4}\s+\d{1,2}:\d{2}\s*WIB.*$', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bFoto\s+Health\b.*$', '', text, flags=re.IGNORECASE)
    
    # 3. Hapus footer promosi, ajakan langganan, download app, & WhatsApp
    footer_patterns = [
        r'KOMPAS\.com berkomitmen.*',
        r'Dukung keberlanjutan jurnalisme.*',
        r'Gabung KOMPAS\.com Plus sekarang.*',
        r'Dapatkan update berita pilihan.*',
        r'Simak berita pilihan lainnya.*',
        r'Ikuti saluran WhatsApp.*',
        r'Tonton video lainnya.*',
        r'Simak video pilihan berikut.*',
        r'Download aplikasi Kompas\.com.*',
        r'Baca artikel menarik lainnya di Google News.*',
        r'ADVERTISEMENT\b.*',
        r'SCROLL TO (?:CONTINUE|RESUME) WITH CONTENT\b.*',
        r'Pilihan redaksi:.*',
        r'Ikuti berita menarik lainnya di.*'
    ]
    for pat in footer_patterns:
        text = re.sub(pat, '', text, flags=re.DOTALL | re.IGNORECASE)
        
    # 4. Hapus inisial jurnalis redaksi (e.g. (akd/akd), (sao/dpy), (kna/up), (mso/mso))
    text = re.sub(r'\([a-zA-Z]{2,5}/[a-zA-Z]{2,5}\)', '', text)
    
    # 5. Hapus baris metadata redaksi (Editor:, Penulis:, Reporter:, Sumber:)
    text = re.sub(r'(?:Penulis|Editor|Reporter|Kontributor)\s*:\s*[^.\n]+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Sumber\s*:\s*(?:Kompas|Tribun|Detik|Antara|Liputan6|CNN)[^.\n]*', '', text, flags=re.IGNORECASE)
    
    # 6. Hapus sisipan tautan 'Baca juga' / 'Baca selengkapnya' / 'Simak juga'
    # 6a. Di dalam tanda kurung: (Baca juga: ...)
    text = re.sub(r'\((?:Baca juga|Baca Juga|Simak juga|Simak Juga|Baca selengkapnya|Lihat juga)[\s:]*[^)]*\)', '', text, flags=re.IGNORECASE)
    # 6b. Di tengah paragraf dengan judul artikel yang direkomendasikan
    text = re.sub(r'(?:Baca juga|Baca Juga|Baca selengkapnya|Simak juga|Simak Juga|Lihat juga)\s*:?\s*[^.?!]+[.?!]?', '', text, flags=re.IGNORECASE)
    # 6c. Sisa kata 'baca juga' yang menggantung di akhir teks/paragraf
    text = re.sub(r'\b(?:baca juga|simak juga|baca selengkapnya|lihat juga)\b.*$', '', text, flags=re.IGNORECASE)
    
    # 7. Hapus prefix nama portal dan dateline kota di awal artikel
    portal_names = r'KOMPAS\.com|Liputan6\.com|Detik\.com|detikHealth|Tribunnews\.com|Tribun[a-zA-Z-]*\.com|CNN Indonesia|Tempo\.co|ANTARA(?:\/[A-Z]+)?|Kumparan|CNBC Indonesia'
    text = re.sub(r'^(?:[A-Za-z\s,\.]*?(?:' + portal_names + r'))\s*[-–—:]+\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^[A-Z][a-zA-Z\s]{1,25}\s+[-–—]+\s+', '', text)
    text = re.sub(r'^[A-Z\s]{2,20}\s*[-–—]+\s+', '', text)
    
    # 8. Ganti penyebutan portal di dalam tubuh kalimat agar lebih netral
    text = re.sub(r'\b(?:Kompas\.com|KOMPAS\.com|Detikcom|detikHealth|Tribunnews|Liputan6\.com)\b', 'artikel ini', text)
    
    # 9. Bersihkan sisa tanda baca menggantung di awal/akhir
    text = re.sub(r'^[-–—:,\s]+', '', text)
    text = re.sub(r'[-–—\s]+$', '', text)
    
    # 10. Normalisasi spasi dan tanda baca ganda
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([,.:;?!])', r'\1', text).strip()
    
    # 11. Huruf kapital di awal kalimat
    if text:
        text = text[0].upper() + text[1:]
    return text

def clean_title_text(title: str) -> str:
    """Membersihkan judul berita dari tag editorial kurung siku dan spasi berlebih."""
    if not isinstance(title, str) or not title.strip():
        return ""
    title = title.replace('\xa0', ' ').replace('\u200b', ' ').strip()
    title = re.sub(r'^\s*\[[^\]]+\]\s*', '', title)  # Hapus tag di awal seperti [POPULER]
    title = re.sub(r'\s+', ' ', title).strip()
    return title

def audit_noise(df: pd.DataFrame, col_name: str = 'isi_berita') -> dict:
    """Memeriksa sisa elemen noise pada suatu kolom teks DataFrame."""
    checks = {
        "Tag Video Detik ([Gambas:...])": r'\[gambas:',
        "Inisial Jurnalis Redaksi (abc/xyz)": r'\([a-zA-Z]{2,5}/[a-zA-Z]{2,5}\)',
        "Sisipan 'Baca juga' / 'Simak juga'": r'(?i)\b(baca\s+juga|simak\s+juga|baca\s+selengkapnya)\b',
        "Header 'Simak Video'": r'(?i)simak\s+video',
        "Metadata Redaksi (Penulis/Editor)": r'(?i)\b(penulis|editor|reporter)\s*:',
        "Dateline Portal di Awal Teks": r'(?i)^[^\n]{0,80}?(kompas\.com|detikhealth|detikcom)\s*[-–—:]+',
        "Footer Promosi / Saluran WhatsApp": r'(?i)(kompas\.com berkomitmen|gabung kompas\.com|saluran whatsapp)'
    }
    results = {}
    for label, pat in checks.items():
        cnt = df[col_name].fillna('').str.contains(pat, regex=True).sum()
        results[label] = int(cnt)
    return results

def run_cleaning():
    print("=" * 80)
    print("       PROSES PEMBERSIHAN DATASET BERITA KESEHATAN (TARGET: 0% NOISE)")
    print("=" * 80)
    
    if not os.path.exists(RAW_DATA_PATH):
        print(f"[ERROR] File dataset mentah tidak ditemukan di: {RAW_DATA_PATH}")
        sys.exit(1)
        
    start_time = time.time()
    print(f"\n[1] Membaca dataset mentah dari:\n    {RAW_DATA_PATH}")
    df_raw = pd.read_csv(RAW_DATA_PATH, sep=CSV_DELIMITER, on_bad_lines='skip')
    total_raw = len(df_raw)
    print(f"    -> Total data terbaca: {total_raw:,} baris.")
    
    # Audit awal
    print("\n[2] Menjalankan Audit Noise Awal (Sebelum Dibersihkan)...")
    audit_before = audit_noise(df_raw, 'isi_berita')
    for label, cnt in audit_before.items():
        pct = (cnt / total_raw) * 100
        print(f"    - {label:<42}: {cnt:>5} baris ({pct:>5.1f}%)")
        
    # Eksekusi Pembersihan
    print("\n[3] Membersihkan seluruh baris teks (Judul & Isi Berita)...")
    df_clean = df_raw.copy()
    df_clean['judul'] = df_clean['judul'].apply(clean_title_text)
    df_clean['isi_berita'] = df_clean['isi_berita'].apply(clean_news_text)
    
    # Tambahkan kolom duplikat clean_text untuk kompatibilitas penuh dengan kode pipeline
    df_clean['judul_clean'] = df_clean['judul']
    df_clean['isi_clean'] = df_clean['isi_berita']
    
    # Filter dokumen kosong atau terlalu pendek (< 100 karakter)
    len_before_filter = len(df_clean)
    df_clean = df_clean.dropna(subset=['judul', 'isi_berita'])
    df_clean = df_clean[df_clean['isi_berita'].str.len() >= 100].reset_index(drop=True)
    dropped_short = len_before_filter - len(df_clean)
    
    # Hapus duplikat isi berita
    len_before_dedup = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=['isi_berita']).reset_index(drop=True)
    dropped_dups = len_before_dedup - len(df_clean)
    
    total_clean = len(df_clean)
    print(f"    -> Dokumen terlalu pendek (<100 char) dibuang: {dropped_short} baris")
    print(f"    -> Dokumen duplikat dibuang                 : {dropped_dups} baris")
    print(f"    -> Total dokumen bersih valid               : {total_clean:,} baris")
    
    # Audit akhir
    print("\n[4] Menjalankan Audit Noise Akhir (Setelah Dibersihkan)...")
    audit_after = audit_noise(df_clean, 'isi_berita')
    all_zero = True
    for label, cnt in audit_after.items():
        pct = (cnt / total_clean) * 100
        status = "[BERSIH 0%]" if cnt == 0 else "[PERINGATAN]"
        print(f"    - {label:<42}: {cnt:>5} baris ({pct:>5.2f}%) {status}")
        if cnt > 0:
            all_zero = False
            
    # Simpan ke CSV baru
    print(f"\n[5] Menyimpan dataset bersih ke file:\n    {CLEAN_DATA_PATH}")
    df_clean.to_csv(CLEAN_DATA_PATH, sep=CSV_DELIMITER, index=False, encoding='utf-8')
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 80)
    if all_zero:
        print("[SUKSES] Seluruh noise berhasil dieliminasi (0% NOISE TERCAPAI).")
    else:
        print("[PERINGATAN] Selesai dengan catatan: Masih ada beberapa sisa noise.")
    print(f"Waktu Proses : {elapsed:.2f} detik")
    print(f"File Hasil   : {CLEAN_DATA_PATH}")
    print(f"Jumlah Baris : {total_clean:,} artikel siap di-indexing.")
    print("=" * 80)

if __name__ == "__main__":
    run_cleaning()
