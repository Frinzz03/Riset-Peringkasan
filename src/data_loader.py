import os
import sys
import pandas as pd

# Pastikan root workspace terdaftar di sys.path agar clean_dataset dapat diimpor
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from clean_dataset import clean_news_text as clean_text, clean_title_text

def load_dataset(file_path: str, delimiter: str = "|") -> pd.DataFrame:
    """
    Membaca dataset CSV berita kesehatan.
    - Jika dataset sudah bersih (hasil clean_dataset.py dengan kolom 'isi_clean'),
      data langsung dimuat secara instan tanpa komputasi ulang.
    - Jika membaca dataset mentah, pembersihan teks otomatis dijalankan.
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
    
    # Cek apakah dataset sudah bersih dari file CSV
    if 'isi_clean' not in df.columns or 'judul_clean' not in df.columns:
        print("Membersihkan teks berita (dataset mentah terdeteksi)...")
        df['judul_clean'] = df['judul'].apply(clean_title_text)
        df['isi_clean'] = df['isi_berita'].apply(clean_text)
        df = df[df['isi_clean'].str.len() >= 100].reset_index(drop=True)
    else:
        print("Dataset terverifikasi bersih (bypass regex, loading instan).")
        
    print(f"Dataset berhasil dimuat: {len(df)} dokumen berita kesehatan valid.")
    return df
