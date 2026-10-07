import pandas as pd
import re

def clean_text(text: str) -> str:
    """
    Membersihkan teks berita kesehatan secara komprehensif dari:
    1. Dateline / prefix nama kota dan portal berita di awal teks (e.g. 'KOMPAS.com - ', 'Jakarta - ').
    2. Sisipan tautan promosi internal ('Baca juga: ...', 'Baca selengkapnya ...', 'Simak juga: ...').
    3. Teks redaksional penutup (Editor, Penulis, ajakan langganan, hak cipta).
    4. Karakter anomali, spasi ganda, dan karakter non-ASCII.
    """
    if not isinstance(text, str):
        return ""
    
    # 0. Normalisasi spasi tak terlihat & karakter khusus
    text = text.replace('\xa0', ' ').replace('\u200b', ' ').replace('\ufeff', ' ').replace('\ufffd', ' ')
    
    # 1. Hapus header & footer tanggal/jam portal (e.g. 'Rabu, 29 Jul 2026 17:43 WIB' dan 'Foto Health ...')
    text = re.sub(r'^[^\n]{0,100}?(?:Senin|Selasa|Rabu|Kamis|Jumat|Sabtu|Minggu),\s*\d{1,2}\s+[A-Za-z]+\s+\d{4}\s+\d{1,2}:\d{2}\s*WIB\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(?:Foto\s+(?:Health|News|Detik)[^\n]*)?(?:Senin|Selasa|Rabu|Kamis|Jumat|Sabtu|Minggu),\s*\d{1,2}\s+[A-Za-z]+\s+\d{4}\s+\d{1,2}:\d{2}\s*WIB.*$', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bFoto\s+Health\b.*$', '', text, flags=re.IGNORECASE)

    # 2. Hapus footer promosi penutup (dari awal pola sampai akhir teks)
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
        r'Baca artikel menarik lainnya di Google News.*'
    ]
    for pat in footer_patterns:
        text = re.sub(pat, '', text, flags=re.DOTALL | re.IGNORECASE)
        
    # 3. Hapus baris metadata redaksi di akhir (Editor:, Penulis:, dsb)
    text = re.sub(r'(?:Penulis|Editor|Reporter|Kontributor)\s*:\s*[^.\n]+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Sumber\s*:\s*(?:Kompas|Tribun|Detik|Antara|Liputan6|CNN)[^.\n]*', '', text, flags=re.IGNORECASE)
    
    # 4. Hapus tautan sisipan 'Baca juga' / 'Baca selengkapnya' (dengan atau tanpa tanda kurung)
    text = re.sub(r'\((?:Baca juga|Baca Juga|Simak juga|Simak Juga|Baca selengkapnya)[\s:]*[^)]*\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(?:Baca juga|Baca Juga|Baca selengkapnya|Simak juga|Simak Juga)[\s:]*[^.?!]+[.?!]?', '', text, flags=re.IGNORECASE)
    
    # 5. Hapus prefix / dateline awal berita (Kota, Nama Portal - )
    portal_names = r'KOMPAS\.com|Liputan6\.com|Detik\.com|detikHealth|Tribunnews\.com|Tribun[a-zA-Z-]*\.com|CNN Indonesia|Tempo\.co|ANTARA(?:\/[A-Z]+)?|Kumparan|CNBC Indonesia'
    text = re.sub(r'^(?:[A-Za-z\s,\.]*?(?:' + portal_names + r'))\s*[-–—:]+\s*', '', text, flags=re.IGNORECASE)
    # Dateline kota: pastikan spasi sebelum strip atau huruf kapital penuh agar kata ulang (seperti 'Baru-baru ini') tidak terpotong
    text = re.sub(r'^[A-Z][a-zA-Z\s]{1,25}\s+[-–—]+\s+', '', text)
    text = re.sub(r'^[A-Z\s]{2,20}\s*[-–—]+\s+', '', text)
    
    # 6. Ganti penyebutan nama portal di dalam kalimat berita
    text = re.sub(r'\b(?:Kompas\.com|KOMPAS\.com|Detikcom|detikHealth|Tribunnews|Liputan6\.com)\b', 'artikel ini', text)
    
    # 7. Normalisasi spasi dan tanda baca ganda
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([,.:;?!])', r'\1', text).strip()
    
    # 8. Pastikan huruf pertama teks selalu kapital
    if text:
        text = text[0].upper() + text[1:]
    return text

def load_dataset(file_path: str, delimiter: str = "|") -> pd.DataFrame:
    """
    Membaca dataset CSV berita kesehatan dan melakukan preprocessing awal.
    """
    print(f"Loading dataset dari: {file_path}")
    df = pd.read_csv(file_path, sep=delimiter, on_bad_lines='skip')
    
    # Pastikan kolom utama tersedia
    required_cols = ['judul', 'isi_berita']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Kolom '{col}' tidak ditemukan dalam dataset. Kolom yang ada: {list(df.columns)}")
    
    # Hapus baris kosong
    df = df.dropna(subset=['judul', 'isi_berita']).copy()
    
    # Pembersihan teks
    print("Membersihkan teks berita...")
    df['judul_clean'] = df['judul'].apply(clean_text)
    df['isi_clean'] = df['isi_berita'].apply(clean_text)
    
    # Hapus berita yang terlalu pendek setelah dibersihkan
    df = df[df['isi_clean'].str.len() > 100].reset_index(drop=True)
    
    print(f"Dataset berhasil dimuat: {len(df)} dokumen berita kesehatan valid.")
    return df
