def build_context_stuffing_prompt(query: str, retrieved_chunks: list) -> str:
    """
    Menggabungkan beberapa dokumen hasil retrieval ke dalam satu prompt konteks terstruktur
    menggunakan strategi Context Stuffing.
    """
    context_str = ""
    for idx, item in enumerate(retrieved_chunks, 1):
        judul = item.get("judul", "Tanpa Judul")
        teks = item.get("chunk_text", "")
        context_str += f"[DOKUMEN {idx}]\n"
        context_str += f"Judul Berita: {judul}\n"
        context_str += f"Potongan Teks: {teks}\n\n"
    
    prompt = f"""Anda adalah sistem peringkasan berita kesehatan berbasis AI. Tugas Anda adalah membaca beberapa dokumen berita kesehatan di bawah ini dan membuat SATU ringkasan multi-dokumen yang koheren, terstruktur, dan akurat.

TOPIK / QUERY PENGGUNA:
"{query}"

DAFTAR DOKUMEN BERITA TERKAIT (CONTEXT STUFFING):
{context_str}

INSTRUKSI PENGRINGKASAN:
1. Gabungkan poin-poin informasi penting dari seluruh DOKUMEN di atas menjadi satu ringkasan multi-dokumen yang utuh.
2. Hindari pengulangan informasi (redundansi) antar berita.
3. Pertahankan fakta utama, gejala, penyebab, obat/solusi, dan statistik penting jika ada.
4. JANGAN menambahkan informasi di luar teks konteks di atas (hindari halusinasi).
5. Tuliskan ringkasan dalam bentuk paragraf terstruktur dan poin-poin penting dalam Bahasa Indonesia yang formal dan mudah dipahami.

RINGKASAN MULTI-DOKUMEN:"""

    return prompt
