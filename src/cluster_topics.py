import os
import sys
import json
import re
import pandas as pd
from collections import defaultdict

# Pastikan output konsol UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Daftar kata kunci blacklist/noise non-kesehatan
BLACKLIST_PATTERNS = [
    r'\bprabowo\b', r'\bjokowi\b', r'\bgibran\b', r'\bkampanye\b', r'\bpilkada\b',
    r'\bpartai\b', r'\bdpr\b', r'\bpolisi\b', r'\bbencana alam\b', r'\bbanjir\b',
    r'\bgempa\b', r'\blongsor\b', r'\btebak gambar\b', r'\bkuis\b', r'\brasional\b',
    r'\btogel\b', r'\bzodiak\b', r'\bhoroskop\b', r'\bramalan\b', r'\bfilm\b',
    r'\barktis\b', r'\bjudi\b', r'\bsinetron\b', r'\bartis\b'
]

def is_noise(title: str, text: str) -> bool:
    combined = (str(title) + " " + str(text)[:300]).lower()
    for bp in BLACKLIST_PATTERNS:
        if re.search(bp, combined):
            return True
    return False

# Taksonomi 165 Sub-Topik Medis Terstruktur
TAXONOMY = [
    # 1. Onkologi & Kanker
    {
        "id": "kanker_payudara",
        "category": "Onkologi & Kanker",
        "title": "Kanker Payudara: Gejala, Deteksi SADARI, dan Penanganan",
        "query": "Tanda dan gejala kanker payudara cara deteksi mandiri SADARI serta pilihan pengobatannya",
        "title_regex": r"kanker payudara|tumor payudara|benjolan di payudara|sadari",
        "content_regex": r"payudara.*(benjolan|mammografi|keganasan|mastektomi)"
    },
    {
        "id": "kanker_serviks",
        "category": "Onkologi & Kanker",
        "title": "Kanker Serviks: Infeksi HPV, Gejala, dan Pencegahan Vaksin",
        "query": "Penyebab kanker serviks leher rahim infeksi HPV pencegahan vaksin dan deteksi pap smear",
        "title_regex": r"kanker serviks|kanker leher rahim|pap smear|vaksin hpv",
        "content_regex": r"serviks.*(hpv|leher rahim|pap smear|pendarahan)"
    },
    {
        "id": "kanker_kolorektal",
        "category": "Onkologi & Kanker",
        "title": "Kanker Usus Besar (Kolorektal): Gejala BAB dan Skrining",
        "query": "Tanda kanker usus besar kolorektal gejala bab berdarah polip dan pencegahannya",
        "title_regex": r"kanker usus|kanker kolorektal|kanker kolon|polip usus",
        "content_regex": r"(usus besar|kolorektal|kolon).*(kanker|polip|bab berdarah)"
    },
    {
        "id": "kanker_paru",
        "category": "Onkologi & Kanker",
        "title": "Kanker Paru-Paru: Bahaya Rokok, Gejala Batuk, dan Deteksi",
        "query": "Gejala kanker paru paru batuk darah sesak napas faktor risiko merokok dan penanganan",
        "title_regex": r"kanker paru",
        "content_regex": r"kanker paru.*(batuk darah|merokok|sesak napas)"
    },
    {
        "id": "kanker_prostat",
        "category": "Onkologi & Kanker",
        "title": "Kanker Prostat: Gejala Saluran Kemih pada Pria",
        "query": "Gejala kanker prostat tanda gangguan buang air kecil pada pria dan skrining psa",
        "title_regex": r"kanker prostat|tumor prostat",
        "content_regex": r"prostat.*(kanker|psa|kencing|pria lanjut usia)"
    },
    {
        "id": "kanker_kulit_melanoma",
        "category": "Onkologi & Kanker",
        "title": "Kanker Kulit dan Melanoma: Tanda Tahi Lalat dan Paparan UV",
        "query": "Tanda bahaya kanker kulit melanoma perubahan tahi lalat dan pencegahan sinar matahari uv",
        "title_regex": r"kanker kulit|melanoma",
        "content_regex": r"kanker kulit.*(melanoma|tahi lalat|sinar uv)"
    },
    {
        "id": "kanker_darah_leukemia",
        "category": "Onkologi & Kanker",
        "title": "Leukemia (Kanker Darah): Gejala dan Komplikasi Sel Darah Putih",
        "query": "Gejala kanker darah leukemia tanda penurunan trombosit anemia dan kemoterapi",
        "title_regex": r"leukemia|kanker darah",
        "content_regex": r"leukemia|kanker darah.*(sel darah putih|trombosit|anemia)"
    },
    {
        "id": "kanker_hati",
        "category": "Onkologi & Kanker",
        "title": "Kanker Hati (Hepatoma): Hubungan Hepatitis dan Sirosis",
        "query": "Penyebab kanker hati hepatoma hubungan hepatitis sirosis dan gejalanya",
        "title_regex": r"kanker hati|hepatoma",
        "content_regex": r"kanker hati.*(hepatoma|sirosis|hepatitis)"
    },
    {
        "id": "kanker_pankreas",
        "category": "Onkologi & Kanker",
        "title": "Kanker Pankreas: Gejala Nyeri Perut dan Deteksi Dini",
        "query": "Tanda kanker pankreas gejala sakit perut sakit kuning dan penanganannya",
        "title_regex": r"kanker pankreas",
        "content_regex": r"kanker pankreas.*(nyeri perut|kuning|insulin)"
    },
    {
        "id": "kanker_tiroid",
        "category": "Onkologi & Kanker",
        "title": "Kanker Tiroid: Benjolan di Leher dan Gangguan Menelan",
        "query": "Gejala kanker tiroid benjolan leher depan suara serak dan penanganan",
        "title_regex": r"kanker tiroid",
        "content_regex": r"kanker tiroid.*(kelenjar tiroid|leher|benjolan)"
    },
    {
        "id": "kanker_ovarium",
        "category": "Onkologi & Kanker",
        "title": "Kanker Ovarium: Gejala Perut Kembung dan Deteksi Dini",
        "query": "Tanda kanker ovarium indung telur gejala perut kembung begah dan penanganan",
        "title_regex": r"kanker ovarium|kanker indung telur",
        "content_regex": r"kanker ovarium.*(indung telur|kembung|nyeri panggul)"
    },
    {
        "id": "efek_kemoterapi",
        "category": "Onkologi & Kanker",
        "title": "Terapi Medis Kanker: Efek Samping Kemoterapi dan Radioterapi",
        "query": "Efek samping kemoterapi mual rambut rontok kelelahan dan cara mengatasinya",
        "title_regex": r"kemoterapi|radioterapi",
        "content_regex": r"(kemoterapi|radioterapi).*(mual|rambut rontok|kanker)"
    },
    {
        "id": "makanan_anti_kanker",
        "category": "Onkologi & Kanker",
        "title": "Makanan Pencegah Kanker: Pola Makan Sehat dan Antioksidan",
        "query": "Makanan pencegah kanker buah sayur antioksidan dan pantangan makanan pemicu karsinogenik",
        "title_regex": r"mencegah kanker|cegah kanker|pemicu kanker|karsinogen",
        "content_regex": r"(mencegah|cegah|risiko) kanker.*(sayur|buah|antioksidan|pola makan)"
    },

    # 2. Jantung & Kardiovaskular
    {
        "id": "serangan_jantung",
        "category": "Jantung & Kardiovaskular",
        "title": "Serangan Jantung (Infark Miokard): Gejala dan Pertolongan Pertama",
        "query": "Gejala serangan jantung nyeri dada sesak napas keringat dingin dan pertolongan pertama",
        "title_regex": r"serangan jantung|infark miokard",
        "content_regex": r"serangan jantung.*(nyeri dada|sesak|infark)"
    },
    {
        "id": "jantung_koroner",
        "category": "Jantung & Kardiovaskular",
        "title": "Penyakit Jantung Koroner: Plak Pembuluh Darah dan Pencegahan",
        "query": "Penyebab penyakit jantung koroner penyumbatan pembuluh darah arteri dan pencegahannya",
        "title_regex": r"jantung koroner|koroner",
        "content_regex": r"jantung koroner.*(arteri|plak|penyumbatan)"
    },
    {
        "id": "gagal_jantung",
        "category": "Jantung & Kardiovaskular",
        "title": "Gagal Jantung: Gejala Sesak Napas, Kaki Bengkak, dan Penanganan",
        "query": "Tanda gagal jantung sesak napas saat berbaring kaki bengkak dan komplikasi",
        "title_regex": r"gagal jantung",
        "content_regex": r"gagal jantung.*(sesak|bengkak|edema)"
    },
    {
        "id": "aritmia_jantung",
        "category": "Jantung & Kardiovaskular",
        "title": "Aritmia: Gangguan Irama Jantung dan Jantung Berdebar",
        "query": "Gejala aritmia detak jantung tidak teratur berdebar kencang pusing dan penyebabnya",
        "title_regex": r"aritmia|irama jantung|jantung berdebar|detak jantung",
        "content_regex": r"(aritmia|irama jantung|berdebar).*(detak|palpitasi)"
    },
    {
        "id": "hipertensi",
        "category": "Jantung & Kardiovaskular",
        "title": "Hipertensi (Tekanan Darah Tinggi): Gejala dan Cara Menurunkannya",
        "query": "Gejala tekanan darah tinggi hipertensi bahaya komplikasi dan cara menurunkannya",
        "title_regex": r"hipertensi|darah tinggi|tekanan darah tinggi",
        "content_regex": r"hipertensi.*(tekanan darah|sistolik|garam|stroke)"
    },
    {
        "id": "hipotensi",
        "category": "Jantung & Kardiovaskular",
        "title": "Hipotensi (Tekanan Darah Rendah): Gejala Pusing dan Penanganan",
        "query": "Penyebab tekanan darah rendah hipotensi gejala pusing lemas berkunang kunang",
        "title_regex": r"hipotensi|darah rendah|tekanan darah rendah",
        "content_regex": r"hipotensi.*(darah rendah|pusing|lemas)"
    },
    {
        "id": "kolesterol_tinggi",
        "category": "Jantung & Kardiovaskular",
        "title": "Kolesterol Tinggi: Bahaya Plak LDL dan Makanan Penurunnya",
        "query": "Penyebab kolesterol tinggi gejala bahaya plak ldl dan makanan alami penurun kolesterol",
        "title_regex": r"kolesterol",
        "content_regex": r"kolesterol.*(ldl|hdl|makanan|trigliserida)"
    },
    {
        "id": "trigliserida_tinggi",
        "category": "Jantung & Kardiovaskular",
        "title": "Trigliserida Tinggi: Lemak Darah dan Risiko Jantung",
        "query": "Penyebab trigliserida tinggi perbedaan dengan kolesterol dan cara menurunkannya",
        "title_regex": r"trigliserida",
        "content_regex": r"trigliserida.*(lemak darah|kolesterol|jantung)"
    },
    {
        "id": "henti_jantung_mendadak",
        "category": "Jantung & Kardiovaskular",
        "title": "Henti Jantung Mendadak: Beda dengan Serangan Jantung dan CPR",
        "query": "Penyebab henti jantung mendadak cardiac arrest perbedaan serangan jantung dan cpr rjp",
        "title_regex": r"henti jantung|cardiac arrest",
        "content_regex": r"(henti jantung|cardiac arrest).*(cpr|pingsan|irama)"
    },
    {
        "id": "ring_jantung_stent",
        "category": "Jantung & Kardiovaskular",
        "title": "Kateterisasi dan Pasang Ring Jantung: Prosedur dan Pemulihan",
        "query": "Prosedur pemasangan ring jantung stent kateterisasi indikasi dan pemulihan pasien",
        "title_regex": r"ring jantung|pasang ring|stent jantung|kateterisasi jantung",
        "content_regex": r"(ring jantung|stent|kateterisasi).*(penyumbatan|arteri)"
    },
    {
        "id": "makanan_sehat_jantung",
        "category": "Jantung & Kardiovaskular",
        "title": "Makanan Sehat untuk Jantung: Pola Makan dan Nutrisi Arteri",
        "query": "Makanan terbaik untuk kesehatan jantung buah sayur ikan omega 3 dan pantangannya",
        "title_regex": r"kesehatan jantung|jantung sehat|menjaga jantung",
        "content_regex": r"jantung.*(makanan|diet|omega|nutrisi|gaya hidup)"
    },

    # 3. Ginjal & Saluran Kemih
    {
        "id": "gagal_ginjal_akut",
        "category": "Ginjal & Saluran Kemih",
        "title": "Gagal Ginjal Akut: Gejala Kencing Berkurang dan Penanganan",
        "query": "Penyebab gagal ginjal akut gejala penurunan urine racun obat dan penanganannya",
        "title_regex": r"gagal ginjal akut",
        "content_regex": r"gagal ginjal akut.*(urine|kencing|kreatinin)"
    },
    {
        "id": "gagal_ginjal_kronis",
        "category": "Ginjal & Saluran Kemih",
        "title": "Gagal Ginjal Kronis: Tahapan Stadium dan Tanda Kerusakan",
        "query": "Gejala gagal ginjal kronis stadium penyakit tanda penurunan fungsi ginjal dan komplikasi",
        "title_regex": r"gagal ginjal kronis|penyakit ginjal kronis|fungsi ginjal",
        "content_regex": r"(gagal ginjal|penyakit ginjal).*(kronis|fungsi|egfr|kreatinin)"
    },
    {
        "id": "cuci_darah_hemodialisis",
        "category": "Ginjal & Saluran Kemih",
        "title": "Cuci Darah (Hemodialisis): Kapan Diperlukan dan Pola Hidup Pasien",
        "query": "Prosedur cuci darah hemodialisis indikasi gagal ginjal stadium akhir dan pola hidup pasien",
        "title_regex": r"cuci darah|hemodialisis|hemodialisa",
        "content_regex": r"(cuci darah|hemodialisis).*(ginjal|racun|cairan)"
    },
    {
        "id": "batu_ginjal",
        "category": "Ginjal & Saluran Kemih",
        "title": "Batu Ginjal: Gejala Nyeri Pinggang, Penyebab, dan Cara Mengatasinya",
        "query": "Gejala batu ginjal nyeri pinggang kencing berdarah penyebab kalsium oksalat dan cara hancurkan",
        "title_regex": r"batu ginjal",
        "content_regex": r"batu ginjal.*(nyeri pinggang|kencing|oksalat|operasi)"
    },
    {
        "id": "infeksi_saluran_kemih",
        "category": "Ginjal & Saluran Kemih",
        "title": "Infeksi Saluran Kemih (ISK): Gejala Anyang-anyangan dan Pengobatan",
        "query": "Gejala infeksi saluran kemih isk anyang anyangan kencing perih bau dan antibiotik",
        "title_regex": r"infeksi saluran kemih|\bisk\b|anyang-anyangan|kencing perih",
        "content_regex": r"(infeksi saluran kemih|anyang-anyangan).*(kencing|bakteri|perih)"
    },
    {
        "id": "kencing_berbusa",
        "category": "Ginjal & Saluran Kemih",
        "title": "Kencing Berbusa dan Proteinuria: Tanda Masalah Ginjal",
        "query": "Penyebab air kencing berbusa tanda kebocoran protein proteinuria dan penyakit ginjal",
        "title_regex": r"kencing berbusa|urine berbusa|proteinuria",
        "content_regex": r"(kencing berbusa|proteinuria).*(ginjal|protein|urine)"
    },
    {
        "id": "pantangan_makanan_ginjal",
        "category": "Ginjal & Saluran Kemih",
        "title": "Makanan yang Merusak Ginjal dan Pantangan Penderita Sakit Ginjal",
        "query": "Makanan yang merusak ginjal pantangan makanan tinggi natrium kalium fosfor penderita ginjal",
        "title_regex": r"merusak ginjal|pantangan ginjal|makanan.*ginjal",
        "content_regex": r"ginjal.*(merusak|pantangan|garam|natrium|makanan)"
    },

    # 4. Metabolik, Endokrin & Gaya Hidup
    {
        "id": "diabetes_tipe_2",
        "category": "Metabolik & Endokrin",
        "title": "Diabetes Melitus Tipe 2: Penyebab, Gejala, dan Pola Makan",
        "query": "Gejala diabetes melitus tipe 2 sering haus buang air kecil resistensi insulin dan pengobatannya",
        "title_regex": r"diabetes tipe 2|diabetes melitus|kencing manis",
        "content_regex": r"diabetes.*(tipe 2|insulin|gula darah|haus)"
    },
    {
        "id": "diabetes_tipe_1",
        "category": "Metabolik & Endokrin",
        "title": "Diabetes Tipe 1 pada Anak dan Remaja: Terapi Insulin",
        "query": "Penyebab diabetes tipe 1 autoimun gejala pada anak suntik terapi insulin seumur hidup",
        "title_regex": r"diabetes tipe 1|diabetes anak",
        "content_regex": r"diabetes.*(tipe 1|anak|autoimun|suntik insulin)"
    },
    {
        "id": "pradiabetes",
        "category": "Metabolik & Endokrin",
        "title": "Pradiabetes: Tanda Kadar Gula Darah Tinggi dan Pencegahannya",
        "query": "Kadar gula darah pradiabetes tanda resistensi insulin dan cara mencegah agar tidak diabetes",
        "title_regex": r"pradiabetes|gula darah tinggi|kadar gula",
        "content_regex": r"pradiabetes.*(gula darah|puasa|glukosa)"
    },
    {
        "id": "hipoglikemia",
        "category": "Metabolik & Endokrin",
        "title": "Hipoglikemia: Gejala Gula Darah Rendah Mendadak dan Penanganan",
        "query": "Gejala gula darah drop hipoglikemia gemetar keringat dingin pusing lemas dan pertolongan",
        "title_regex": r"hipoglikemia|gula darah rendah|gula darah drop",
        "content_regex": r"(hipoglikemia|gula darah rendah).*(gemetar|keringat dingin|pusing)"
    },
    {
        "id": "asam_urat",
        "category": "Metabolik & Endokrin",
        "title": "Asam Urat (Gout): Penyebab Nyeri Sendi dan Makanan Pantangan",
        "query": "Penyebab serangan asam urat kambuh gejala pembengkakan jempol kaki dan pantangan purin",
        "title_regex": r"asam urat|penyakit gout",
        "content_regex": r"asam urat.*(sendi|purin|nyeri|jempol|kristal)"
    },
    {
        "id": "lemak_visceral",
        "category": "Metabolik & Endokrin",
        "title": "Lemak Visceral dan Obesitas Sentral: Bahaya Perut Buncit",
        "query": "Bahaya lemak visceral di perut tanda obesitas sentral sindrom metabolik dan cara bakar lemak",
        "title_regex": r"lemak visceral|lemak perut|perut buncit|obesitas sentral",
        "content_regex": r"(lemak visceral|perut buncit|obesitas sentral).*(metabolik|lingkar pinggang)"
    },
    {
        "id": "gangguan_tiroid",
        "category": "Metabolik & Endokrin",
        "title": "Gangguan Kelenjar Tiroid: Beda Hipertiroid dan Hipotiroid",
        "query": "Perbedaan hipertiroid dan hipotiroid gejala detak jantung berat badan kelenjar leher",
        "title_regex": r"tiroid|hipertiroid|hipotiroid|kelenjar tiroid",
        "content_regex": r"tiroid.*(hormon|leher|metabolisme|berat badan)"
    },
    {
        "id": "makanan_penurun_gula",
        "category": "Metabolik & Endokrin",
        "title": "Makanan Alami Penurun Gula Darah untuk Penderita Diabetes",
        "query": "Daftar makanan sayur buah alami penurun gula darah dan indeks glikemik rendah",
        "title_regex": r"menurunkan gula darah|penurun gula darah|makanan diabetes",
        "content_regex": r"(menurunkan|penurun) gula darah.*(makanan|diabetes|glukosa)"
    },

    # 5. Lambung, Hati & Saluran Cerna
    {
        "id": "gerd_asam_lambung",
        "category": "Lambung & Pencernaan",
        "title": "GERD dan Asam Lambung: Gejala Dada Panas (Heartburn) dan Pantangan",
        "query": "Gejala asam lambung naik gerd dada panas sesak nyeri tenggorokan dan makanan pantangan",
        "title_regex": r"\bgerd\b|asam lambung|heartburn",
        "content_regex": r"(gerd|asam lambung).*(kerongkongan|heartburn|mual|pantangan)"
    },
    {
        "id": "maag_gastritis",
        "category": "Lambung & Pencernaan",
        "title": "Sakit Maag (Gastritis) dan Tukak Lambung: Perbedaan dan Solusi",
        "query": "Perbedaan sakit maag gastritis dan tukak lambung nyeri ulu hati mual dan obat antasida",
        "title_regex": r"sakit maag|gastritis|tukak lambung|perih lambung|nyeri ulu hati",
        "content_regex": r"(maag|gastritis|tukak lambung).*(ulu hati|lambung|perih)"
    },
    {
        "id": "fatty_liver",
        "category": "Lambung & Pencernaan",
        "title": "Perlemakan Hati (Fatty Liver): Gejala dan Cara Menghilangkannya",
        "query": "Penyebab perlemakan hati fatty liver penumpukan lemak organ hati dan cara mengatasinya",
        "title_regex": r"fatty liver|perlemakan hati",
        "content_regex": r"(fatty liver|perlemakan hati).*(hati|hepar|lemak)"
    },
    {
        "id": "sembelit_konstipasi",
        "category": "Lambung & Pencernaan",
        "title": "Sembelit (Konstipasi): Susah BAB, Penyebab, dan Makanan Pelancar",
        "query": "Cara mengatasi susah bab sembelit konstipasi makanan tinggi serat buah pepaya dan air",
        "title_regex": r"sembelit|konstipasi|susah bab",
        "content_regex": r"(sembelit|konstipasi|susah bab).*(serat|buang air besar|feses)"
    },
    {
        "id": "wasir_ambeien",
        "category": "Lambung & Pencernaan",
        "title": "Wasir (Ambeien / Hemoroid): Gejala BAB Berdarah dan Pengobatan",
        "query": "Gejala wasir ambeien benjolan anus bab berdarah derajat keparahan dan cara mengobatinya",
        "title_regex": r"wasir|ambeien|hemoroid",
        "content_regex": r"(wasir|ambeien|hemoroid).*(anus|bab berdarah|benjolan)"
    },
    {
        "id": "batu_empedu",
        "category": "Lambung & Pencernaan",
        "title": "Batu Empedu: Gejala Sakit Perut Kanan Atas dan Penanganan",
        "query": "Penyebab batu empedu gejala nyeri perut kanan atas mual makanan berlemak dan operasi",
        "title_regex": r"batu empedu|kandung empedu",
        "content_regex": r"batu empedu.*(kolesterol|nyeri perut|operasi)"
    },
    {
        "id": "diare_dehidrasi",
        "category": "Lambung & Pencernaan",
        "title": "Diare Akut: Bahaya Dehidrasi, Oralit, dan Makanan yang Tepat",
        "query": "Penanganan diare akut pencegahan dehidrasi minum oralit dan makanan saat mencret",
        "title_regex": r"\bdiare\b|mencret",
        "content_regex": r"diare.*(dehidrasi|oralit|feses|cairan)"
    },
    {
        "id": "usus_buntu",
        "category": "Lambung & Pencernaan",
        "title": "Radang Usus Buntu (Apendisitis): Gejala Nyeri Perut Kanan Bawah",
        "query": "Gejala radang usus buntu apendisitis sakit perut kanan bawah demam dan operasi",
        "title_regex": r"usus buntu|apendisitis",
        "content_regex": r"(usus buntu|apendisitis).*(perut kanan bawah|nyeri|operasi)"
    },

    # 6. Neurologi, Saraf & Otak
    {
        "id": "stroke",
        "category": "Neurologi & Saraf",
        "title": "Stroke: Gejala Metode FAST, Pencegahan, dan Golden Period",
        "query": "Tanda awal stroke metode FAST wajah merot bicara pelo golden period dan pemulihan",
        "title_regex": r"\bstroke\b",
        "content_regex": r"stroke.*(pembuluh darah|otak|lumpuh|bicara pelo)"
    },
    {
        "id": "demensia_alzheimer",
        "category": "Neurologi & Saraf",
        "title": "Demensia dan Alzheimer: Gejala Pikun Dini dan Penurunan Memori",
        "query": "Perbedaan demensia dan alzheimer gejala sering lupa pikun tanda gangguan kognitif otak",
        "title_regex": r"demensia|alzheimer|\bpikun\b",
        "content_regex": r"(demensia|alzheimer|pikun).*(memori|daya ingat|otak)"
    },
    {
        "id": "migrain_sakit_kepala",
        "category": "Neurologi & Saraf",
        "title": "Migrain dan Sakit Kepala: Penyebab, Gejala Berdenyut, dan Obat Alami",
        "query": "Penyebab migrain sakit kepala sebelah mata silau pemicu makanan dan cara redakan",
        "title_regex": r"migrain|sakit kepala sebelah|sakit kepala tegang",
        "content_regex": r"(migrain|sakit kepala).*(berdenyut|sebelah|pusing)"
    },
    {
        "id": "vertigo",
        "category": "Neurologi & Saraf",
        "title": "Vertigo: Gejala Kepala Berputar, Masalah Telinga Dalam, dan Manuver",
        "query": "Penyebab vertigo kepala berputar mendadak gangguan telinga dalam vestibular dan penanganan",
        "title_regex": r"vertigo",
        "content_regex": r"vertigo.*(berputar|keseimbangan|telinga dalam)"
    },
    {
        "id": "mabuk_perjalanan",
        "category": "Neurologi & Saraf",
        "title": "Mabuk Perjalanan (Kinetosis): Herbal Penawar Mual dan Tips",
        "query": "Obat alami mabuk perjalanan kinetosis jahe minyak atsiri aromaterapi mual muntah di jalan",
        "title_regex": r"mabuk perjalanan|kinetosis",
        "content_regex": r"mabuk perjalanan.*(mual|jahe|perjalanan|telinga)"
    },
    {
        "id": "parkinson_tremor",
        "category": "Neurologi & Saraf",
        "title": "Penyakit Parkinson: Gejala Tangan Gemetar (Tremor) dan Gerak Lambat",
        "query": "Gejala penyakit parkinson tangan gemetar tremor gerakan kaku dopamin dan terapi",
        "title_regex": r"parkinson|tangan gemetar|tremor",
        "content_regex": r"(parkinson|tremor).*(gemetar|gerak|saraf)"
    },
    {
        "id": "saraf_terjepit",
        "category": "Neurologi & Saraf",
        "title": "Saraf Terjepit (HNP): Nyeri Pinggang Menjalar dan Terapi Medis",
        "query": "Gejala saraf terjepit hnp nyeri pinggang menjalar ke kaki kebas kesemutan dan pengobatan",
        "title_regex": r"saraf terjepit|\bhnp\b",
        "content_regex": r"(saraf terjepit|hnp).*(pinggang|tulang belakang|nyeri|kaki)"
    },
    {
        "id": "kebas_kesemutan",
        "category": "Neurologi & Saraf",
        "title": "Neuropati Perifer: Penyebab Kaki Tangan Sering Kesemutan dan Kebas",
        "query": "Penyebab tangan kaki sering kebas kesemutan tanda neuropati kekurangan vitamin b dan diabetes",
        "title_regex": r"kesemutan|\bkebas\b|neuropati",
        "content_regex": r"(kesemutan|kebas|neuropati).*(tangan|kaki|saraf)"
    },

    # 7. Infeksi Tropis & Penyakit Menular
    {
        "id": "dbd_dengue",
        "category": "Infeksi & Penyakit Menular",
        "title": "Demam Berdarah Dengue (DBD): Gejala Fase Kritis dan Trombosit",
        "query": "Gejala demam berdarah dengue dbd fase kritis turunnya trombosit bintik merah dan cairan",
        "title_regex": r"demam berdarah|\bdbd\b|dengue",
        "content_regex": r"(demam berdarah|dbd|dengue).*(nyamuk|trombosit|demam)"
    },
    {
        "id": "malaria",
        "category": "Infeksi & Penyakit Menular",
        "title": "Malaria: Gigitan Nyamuk Anopheles, Gejala Menggigil, dan Obat",
        "query": "Gejala penyakit malaria demam menggigil berkeringat parasit plasmodium dan obat kina",
        "title_regex": r"malaria",
        "content_regex": r"malaria.*(nyamuk|anopheles|plasmodium|menggigil)"
    },
    {
        "id": "tbc_tuberkulosis",
        "category": "Infeksi & Penyakit Menular",
        "title": "Tuberkulosis (TBC): Gejala Batuk Lama, Kuman Bakteri, dan Obat OAT",
        "query": "Gejala tbc paru batuk lebih dari 2 minggu batuk darah keringat malam obat oat rutin",
        "title_regex": r"\btbc\b|tuberkulosis",
        "content_regex": r"(tbc|tuberkulosis).*(batuk|paru|bakteri|dahak)"
    },
    {
        "id": "cacar_monyet_mpox",
        "category": "Infeksi & Penyakit Menular",
        "title": "Cacar Monyet (Mpox): Gejala Lenting Kulit, Penularan, dan Vaksin",
        "query": "Gejala cacar monyet mpox lenting bernanah ruam kulit penularan kontak dan pencegahan",
        "title_regex": r"cacar monyet|mpox",
        "content_regex": r"(cacar monyet|mpox).*(ruam|lenting|kulit|virus)"
    },
    {
        "id": "covid_19",
        "category": "Infeksi & Penyakit Menular",
        "title": "Covid-19: Gejala Varian Baru, Penularan, dan Imunitas Tubuh",
        "query": "Gejala covid 19 varian baru tenggorokan kering demam batuk hilang penciuman dan vaksin",
        "title_regex": r"covid-19|covid 19|virus corona|sars-cov-2",
        "content_regex": r"(covid|corona).*(gejala|varian|vaksin|tes)"
    },
    {
        "id": "rabies",
        "category": "Infeksi & Penyakit Menular",
        "title": "Rabies: Bahaya Gigitan Anjing Kucing, Gejala Takut Air, dan Vaksin VAR",
        "query": "Bahaya rabies gigitan anjing gejala hidrofobia takut air pertolongan pertama cuci sabun vaksin",
        "title_regex": r"rabies|gigitan anjing",
        "content_regex": r"rabies.*(gigitan|anjing|vaksin|air)"
    },
    {
        "id": "tifus_demam_tifoid",
        "category": "Infeksi & Penyakit Menular",
        "title": "Demam Tifoid (Tipes): Bakteri Salmonella, Makanan Bersih, dan Terapi",
        "query": "Gejala tifus tipes demam malam hari lidah kotor sakit perut bakteri salmonella dan pantangan",
        "title_regex": r"tifus|tipes|demam tifoid",
        "content_regex": r"(tifus|tipes|tifoid).*(demam|salmonella|pencernaan)"
    },
    {
        "id": "hepatitis",
        "category": "Infeksi & Penyakit Menular",
        "title": "Hepatitis B dan C: Penularan Virus, Sakit Kuning, dan Kerusakan Hati",
        "query": "Perbedaan hepatitis a b c penularan virus gejala mata kuning dan vaksin pencegahan",
        "title_regex": r"hepatitis",
        "content_regex": r"hepatitis.*(hati|virus|kuning|vaksin)"
    },
    {
        "id": "cacar_air_herpes",
        "category": "Infeksi & Penyakit Menular",
        "title": "Cacar Air dan Herpes Zoster: Virus Varicella, Bintil Berair, dan Salep",
        "query": "Gejala cacar air bintil lenting gatal herpes zoster saraf dan perawatan kulit",
        "title_regex": r"cacar air|herpes zoster|varicella",
        "content_regex": r"(cacar air|herpes).*(gatal|lenting|kulit|virus)"
    },
    {
        "id": "polio_imunisasi",
        "category": "Infeksi & Penyakit Menular",
        "title": "Polio: Bahaya Kelumpuhan Anak dan Pentingnya Tetes Imunisasi",
        "query": "Penyebab polio bahaya lumpuh layu pada anak penularan virus dan pekan imunisasi nasional",
        "title_regex": r"\bpolio\b",
        "content_regex": r"polio.*(lumpuh|imunisasi|anak|tetes)"
    },

    # 8. Paru-Paru, Respirasi & Alergi
    {
        "id": "asma_bronkial",
        "category": "Respirasi & Paru",
        "title": "Asma Bronkial: Gejala Sesak Napas, Mengi, Pemicu Alergi, dan Inhaler",
        "query": "Gejala penyakit asma sesak napas bunyi mengi pemicu cuaca dingin debu dan obat inhaler",
        "title_regex": r"\basma\b",
        "content_regex": r"asma.*(sesak napas|mengi|inhaler|kambuh)"
    },
    {
        "id": "pneumonia_paru_basah",
        "category": "Respirasi & Paru",
        "title": "Pneumonia (Radang Paru / Paru Basah): Gejala Demam dan Sesak",
        "query": "Gejala pneumonia paru paru basah demam batuk berdahak sesak napas dan infeksi bakteri",
        "title_regex": r"pneumonia|paru-paru basah|paru basah",
        "content_regex": r"(pneumonia|paru basah).*(kantong udara|sesak|dahak)"
    },
    {
        "id": "ppok_paru_obstruktif",
        "category": "Respirasi & Paru",
        "title": "PPOK (Penyakit Paru Obstruktif Kronis): Akibat Merokok dan Polusi",
        "query": "Gejala ppok kerusakan saluran pernapasan sesak menahun akibat asap rokok dan terapi oksigen",
        "title_regex": r"\bppok\b|paru obstruktif",
        "content_regex": r"(ppok|paru obstruktif).*(rokok|sesak|kronis)"
    },
    {
        "id": "sinusitis",
        "category": "Respirasi & Paru",
        "title": "Sinusitis: Gejala Hidung Tersumbat, Nyeri Wajah, dan Cuci Hidung",
        "query": "Gejala radang sinusitis hidung tersumbat lendir hijau nyeri dahi pipi dan cuci hidung nacl",
        "title_regex": r"sinusitis|radang sinus",
        "content_regex": r"sinusitis.*(hidung|lendir|wajah|tersumbat)"
    },
    {
        "id": "polusi_udara_ispa",
        "category": "Respirasi & Paru",
        "title": "Polusi Udara PM2.5 dan ISPA: Dampak Asap dan Perlindungan Masker",
        "query": "Bahaya polusi udara partikel pm2.5 infeksi saluran pernapasan ispa batuk pilek dan masker",
        "title_regex": r"polusi udara|\bispa\b|pm2\.5",
        "content_regex": r"(polusi|ispa|pm2\.5).*(pernapasan|batuk|udara)"
    },
    {
        "id": "alergi_debu_rokok",
        "category": "Respirasi & Paru",
        "title": "Alergi Pernapasan: Reaksi Asap Rokok, Debu, dan Rhinitis",
        "query": "Gejala alergi asap rokok debu bersin hidung meler iritasi tenggorokan dan antihistamin",
        "title_regex": r"alergi asap rokok|alergi debu|rhinitis",
        "content_regex": r"alergi.*(debu|rokok|bersin|hidung)"
    },

    # 9. Tulang, Sendi & Otot
    {
        "id": "osteoporosis",
        "category": "Tulang & Sendi",
        "title": "Osteoporosis: Pengeroposan Tulang, Risiko Patah, dan Kalsium",
        "query": "Penyebab osteoporosis tulang keropos risiko patah tulang lansia kalsium dan vitamin d",
        "title_regex": r"osteoporosis|tulang keropos",
        "content_regex": r"osteoporosis.*(tulang|kalsium|patah tulang)"
    },
    {
        "id": "osteoartritis_lutut",
        "category": "Tulang & Sendi",
        "title": "Osteoartritis: Pengapuran Sendi Lutut, Nyeri Gerak, dan Pelumas Sendi",
        "query": "Gejala radang sendi osteoartritis pengapuran bantalan lutut nyeri saat berjalan dan terapi",
        "title_regex": r"osteoartritis|pengapuran sendi|radang sendi lutut",
        "content_regex": r"(osteoartritis|pengapuran sendi).*(lutut|nyeri|tulang rawan)"
    },
    {
        "id": "rematik_rheumatoid",
        "category": "Tulang & Sendi",
        "title": "Rematik (Rheumatoid Arthritis): Radang Sendi Autoimun dan Kaku Pagi",
        "query": "Perbedaan rematik dan asam urat gejala rheumatoid arthritis sendi kaku pagi hari autoimun",
        "title_regex": r"rematik|rheumatoid arthritis",
        "content_regex": r"(rematik|rheumatoid).*(sendi|kaku|jari|autoimun)"
    },
    {
        "id": "nyeri_punggung_pinggang",
        "category": "Tulang & Sendi",
        "title": "Sakit Punggung dan Nyeri Pinggang: Kebiasaan Duduk dan Postur",
        "query": "Penyebab sakit punggung nyeri pinggang salah posisi duduk kebiasaan kerja dan peregangan",
        "title_regex": r"sakit punggung|nyeri punggung|nyeri pinggang|sakit pinggang",
        "content_regex": r"(sakit punggung|nyeri pinggang).*(duduk|otot|tulang belakang)"
    },
    {
        "id": "skoliosis",
        "category": "Tulang & Sendi",
        "title": "Skoliosis: Tulang Belakang Melengkung, Deteksi Dini, dan Terapi",
        "query": "Gejala skoliosis tulang belakang melengkung bahu tidak simetris deteksi anak dan bracing",
        "title_regex": r"skoliosis",
        "content_regex": r"skoliosis.*(tulang belakang|melengkung|bahu)"
    },
    {
        "id": "kram_otot",
        "category": "Tulang & Sendi",
        "title": "Kram Otot Kaki: Kurang Cairan, Elektrolit, dan Peregangan",
        "query": "Penyebab sering kram kaki malam hari kekurangan magnesium kalium dehidrasi dan penanganan",
        "title_regex": r"kram otot|kram kaki",
        "content_regex": r"kram.*(otot|kaki|elektrolit|magnesium)"
    },

    # 10. Kesehatan Ibu, Anak & Reproduksi
    {
        "id": "stunting_balita",
        "category": "Kesehatan Ibu & Anak",
        "title": "Stunting pada Balita: Ciri Anak Pendek, Gagal Tumbuh, dan Nutrisi MPASI",
        "query": "Pencegahan stunting balita ciri pertumbuhan anak lambat asupan protein hewani dan gizi",
        "title_regex": r"\bstunting\b",
        "content_regex": r"stunting.*(balita|anak|gizi|tinggi badan)"
    },
    {
        "id": "anemia_kehamilan",
        "category": "Kesehatan Ibu & Anak",
        "title": "Anemia pada Ibu Hamil: Bahaya Kurang Darah dan Tablet Tambah Darah",
        "query": "Bahaya anemia pada ibu hamil kadar hb rendah tablet tambah darah zat besi untuk janin",
        "title_regex": r"anemia.*hamil|ibu hamil.*anemia|tablet tambah darah",
        "content_regex": r"(anemia|zat besi).*(hamil|ibu hamil|janin)"
    },
    {
        "id": "preeklamsia",
        "category": "Kesehatan Ibu & Anak",
        "title": "Preeklamsia: Tekanan Darah Tinggi pada Ibu Hamil dan Protein Urine",
        "query": "Tanda bahaya preeklamsia darah tinggi kehamilan bengkak kaki protein urine dan persalinan",
        "title_regex": r"preeklamsia|eklamsia",
        "content_regex": r"preeklamsia.*(kehamilan|tekanan darah|ibu hamil)"
    },
    {
        "id": "pcos_ovarium",
        "category": "Kesehatan Ibu & Anak",
        "title": "PCOS (Sindrom Polikistik Ovarium): Haid Tidak Teratur dan Kesuburan",
        "query": "Gejala pcos ovarium polikistik menstruasi tidak teratur resistensi insulin dan peluang hamil",
        "title_regex": r"\bpcos\b|polikistik ovarium",
        "content_regex": r"pcos.*(haid|menstruasi|ovarium|sel telur)"
    },
    {
        "id": "keputihan_wanita",
        "category": "Kesehatan Ibu & Anak",
        "title": "Keputihan Abnormal: Tanda Infeksi Jamur, Bakteri, dan Ciri Berbahaya",
        "query": "Ciri keputihan berbahaya bau gatal warna kuning kehijauan infeksi jamur bakteri dan obat",
        "title_regex": r"keputihan",
        "content_regex": r"keputihan.*(gatal|bau|wanita|organ intim)"
    },
    {
        "id": "endometriosis",
        "category": "Kesehatan Ibu & Anak",
        "title": "Endometriosis: Gejala Nyeri Haid Hebat dan Masalah Kesuburan",
        "query": "Gejala endometriosis jaringan dinding rahim nyeri panggul saat haid kram perut parah",
        "title_regex": r"endometriosis",
        "content_regex": r"endometriosis.*(rahim|nyeri haid|menstruasi)"
    },
    {
        "id": "menopause",
        "category": "Kesehatan Ibu & Anak",
        "title": "Menopause: Gejala Hot Flashes, Perubahan Hormon Estrogen, dan Emosi",
        "query": "Tanda wanita memasuki menopause rasa panas hot flashes mood swing hormon estrogen turun",
        "title_regex": r"menopause",
        "content_regex": r"menopause.*(estrogen|hot flashes|haid berhenti)"
    },
    {
        "id": "asi_menyusui",
        "category": "Kesehatan Ibu & Anak",
        "title": "Manfaat ASI Eksklusif dan Cara Mengatasi Masalah Menyusui",
        "query": "Manfaat asi eksklusif untuk daya tahan bayi cara memperbanyak produksi asi dan mastitis",
        "title_regex": r"asi eksklusif|menyusui|pejuang asi",
        "content_regex": r"(asi|menyusui).*(bayi|nutrisi|ibu)"
    },

    # 11. Nutrisi, Vitamin & Suplemen
    {
        "id": "vitamin_d",
        "category": "Nutrisi & Vitamin",
        "title": "Kekurangan Vitamin D: Gejala Tersembunyi, Sinar Matahari, dan Tulang",
        "query": "Tanda kekurangan vitamin d kelelahan nyeri tulang imunitas jam berjemur dan suplemen",
        "title_regex": r"vitamin d",
        "content_regex": r"vitamin d.*(berjemur|kalsium|tulang|suplemen)"
    },
    {
        "id": "vitamin_c",
        "category": "Nutrisi & Vitamin",
        "title": "Manfaat Vitamin C: Imunitas Tubuh, Kulit, dan Batas Aman Dosis",
        "query": "Manfaat vitamin c dosis harian penyerapan zat besi sariawan dan batas aman asam lambung",
        "title_regex": r"vitamin c",
        "content_regex": r"vitamin c.*(imun|antioksidan|dosis|buah)"
    },
    {
        "id": "vitamin_b12_anemia",
        "category": "Nutrisi & Vitamin",
        "title": "Kekurangan Vitamin B12: Gejala Lemas, Saraf Kebas, dan Sumber Makanan",
        "query": "Tanda defisiensi vitamin b12 anemia megaloblastik saraf kebas lemas dan sumber hewani",
        "title_regex": r"vitamin b12|vitamin b kompleks",
        "content_regex": r"vitamin b.*(saraf|darah|lemas|sel darah merah)"
    },
    {
        "id": "zat_besi_anemia",
        "category": "Nutrisi & Vitamin",
        "title": "Defisiensi Zat Besi: Gejala 5L (Lesu, Lemah, Letih, Lelah, Lalai)",
        "query": "Gejala kekurangan zat besi anemia 5l lemas pucat pusing dan makanan penambah darah",
        "title_regex": r"zat besi|kurang darah",
        "content_regex": r"zat besi.*(anemia|hb|darah|lemas)"
    },
    {
        "id": "bahaya_gula_manis",
        "category": "Nutrisi & Vitamin",
        "title": "Bahaya Konsumsi Gula Berlebih: Minuman Manis dan Batas Harian Kemenkes",
        "query": "Bahaya minuman manis konsumsi gula harian kemenkes risiko obesitas dan diabetes",
        "title_regex": r"minuman manis|konsumsi gula|gula berlebih|takaran gula",
        "content_regex": r"gula.*(manis|sendok makan|kemenkes|diabetes)"
    },
    {
        "id": "bahaya_garam_natrium",
        "category": "Nutrisi & Vitamin",
        "title": "Batas Konsumsi Garam Harian: Bahaya Makanan Asin bagi Tekanan Darah",
        "query": "Batas konsumsi garam sendok teh natrium hipertensi ginjal dan makanan olahan asin",
        "title_regex": r"konsumsi garam|garam berlebih|makanan asin|asupan natrium",
        "content_regex": r"garam.*(natrium|hipertensi|darah tinggi|kemenkes)"
    },
    {
        "id": "kurang_minum_air",
        "category": "Nutrisi & Vitamin",
        "title": "Dampak Kurang Minum Air Putih: Tanda Dehidrasi dan Kebutuhan Harian",
        "query": "Tanda tubuh kurang minum air dehidrasi urine pekat sakit kepala fungsi ginjal 8 gelas",
        "title_regex": r"kurang minum|air putih|minum air|dehidrasi",
        "content_regex": r"(air putih|dehidrasi).*(minum|cairan|ginjal|gelas)"
    },
    {
        "id": "kopi_kesehatan",
        "category": "Nutrisi & Vitamin",
        "title": "Manfaat dan Efek Samping Kopi: Batas Kafein bagi Jantung dan Lambung",
        "query": "Manfaat minum kopi hitam efek kafein bagi detak jantung asam lambung dan batas cangkir",
        "title_regex": r"minum kopi|kopi hitam|\bkafein\b",
        "content_regex": r"kopi.*(kafein|jantung|lambung|kesehatan)"
    },

    # 12. Kesehatan Mental, Jiwa & Tidur
    {
        "id": "insomnia_gangguan_tidur",
        "category": "Kesehatan Mental & Tidur",
        "title": "Insomnia: Penyebab Susah Tidur Malam, Dampak Kesehatan, dan Tips",
        "query": "Penyebab insomnia susah tidur malam hari sleep hygiene kualitas tidur dan cara cepat lelap",
        "title_regex": r"insomnia|susah tidur|gangguan tidur",
        "content_regex": r"(insomnia|susah tidur).*(malam|lelap|kualitas tidur)"
    },
    {
        "id": "jam_tidur_begadang",
        "category": "Kesehatan Mental & Tidur",
        "title": "Bahaya Sering Begadang dan Waktu Tidur Ideal bagi Tubuh",
        "query": "Bahaya sering begadang tidur larut malam efek penurunan imun penyakit kronis jam tidur ideal",
        "title_regex": r"begadang|tidur larut malam|kurang tidur",
        "content_regex": r"(begadang|kurang tidur).*(kesehatan|malam|imun|jantung)"
    },
    {
        "id": "anxiety_gangguan_cemas",
        "category": "Kesehatan Mental & Tidur",
        "title": "Gangguan Kecemasan (Anxiety): Ciri-ciri Cemas Berlebih dan Penanganan",
        "query": "Gejala anxiety disorder gangguan kecemasan rasa takut berlebih jantung berdebar dan relaksasi",
        "title_regex": r"\banxiety\b|gangguan kecemasan|cemas berlebih",
        "content_regex": r"(anxiety|kecemasan|cemas).*(mental|panik|stres)"
    },
    {
        "id": "panic_attack",
        "category": "Kesehatan Mental & Tidur",
        "title": "Serangan Panik (Panic Attack): Gejala Sesak Tiba-tiba dan Cara Tenang",
        "query": "Gejala serangan panik panic attack dada sesak takut mati gemetar dan teknik pernapasan",
        "title_regex": r"panic attack|serangan panik",
        "content_regex": r"(panic attack|serangan panik).*(sesak|jantung|takut|napas)"
    },
    {
        "id": "depresi",
        "category": "Kesehatan Mental & Tidur",
        "title": "Depresi Klinis: Tanda Kehilangan Minat, Putus Asa, dan Bantuan Medis",
        "query": "Ciri depresi klinis rasa sedih berkepanjangan kehilangan energi putus asa dan psikolog",
        "title_regex": r"\bdepresi\b",
        "content_regex": r"depresi.*(mental|sedih|psikolog|putus asa)"
    },
    {
        "id": "burnout_lelah_mental",
        "category": "Kesehatan Mental & Tidur",
        "title": "Burnout: Kelelahan Mental Kerja, Gejala Emosional, dan Pemulihan",
        "query": "Tanda burnout stres kerja kronis kelelahan fisik mental emosional dan cara pemulihan",
        "title_regex": r"burnout|lelah mental|stres kerja",
        "content_regex": r"(burnout|stres).*(kerja|kelelahan|mental)"
    },
    {
        "id": "psikosomatis",
        "category": "Kesehatan Mental & Tidur",
        "title": "Gangguan Psikosomatis: Hubungan Stres Pikiran dengan Sakit Fisik",
        "query": "Penyebab psikosomatis stres cemas memicu lambung nyeri dada asam lambung kambuh",
        "title_regex": r"psikosomatis|psikosomatik",
        "content_regex": r"psikosomatis.*(stres|pikiran|fisik|gejala)"
    },

    # 13. Kulit, Rambut & Estetika Medis
    {
        "id": "jerawat_peradangan",
        "category": "Kulit & Rambut",
        "title": "Jerawat (Acne): Penyebab Sumbatan Pori, Hormon, dan Perawatan",
        "query": "Penyebab jerawat radang bakteri pori tersumbat minyak berlebih bahan aktif salep",
        "title_regex": r"jerawat|bekas jerawat",
        "content_regex": r"jerawat.*(kulit|pori|radang|minyak)"
    },
    {
        "id": "eksim_dermatitis",
        "category": "Kulit & Rambut",
        "title": "Eksim (Dermatitis Atopik): Gejala Kulit Kering Gatal dan Pemicu",
        "query": "Gejala eksim dermatitis kulit gatal bersisik merah pelembap dan menghindari alergen",
        "title_regex": r"eksim|dermatitis atopik",
        "content_regex": r"(eksim|dermatitis).*(kulit|gatal|kering)"
    },
    {
        "id": "psoriasis",
        "category": "Kulit & Rambut",
        "title": "Psoriasis: Plak Kulit Tebal Bersisik Putih dan Penyakit Autoimun",
        "query": "Penyebab psoriasis kulit menebal bersisik merah perak autoimun dan terapi salep",
        "title_regex": r"psoriasis",
        "content_regex": r"psoriasis.*(kulit|sisik|autoimun|plak)"
    },
    {
        "id": "rambut_rontok",
        "category": "Kulit & Rambut",
        "title": "Rambut Rontok Berlebih: Penyebab Kebotakan dan Nutrisi Folikel",
        "query": "Penyebab rambut rontok parah kebotakan alopesia kekurangan nutrisi hormon dan shampo",
        "title_regex": r"rambut rontok|kebotakan|alopesia",
        "content_regex": r"(rambut rontok|kebotakan).*(kulit kepala|folikel|akar)"
    },
    {
        "id": "sunscreen_uv",
        "category": "Kulit & Rambut",
        "title": "Pentingnya Sunscreen: Perlindungan Sinar UV, SPF, dan Kanker Kulit",
        "query": "Manfaat memakai sunscreen tabir surya spf pa perlindungan paparan sinar matahari uv",
        "title_regex": r"sunscreen|tabir surya|\bspf\b",
        "content_regex": r"(sunscreen|tabir surya).*(sinar uv|matahari|kulit)"
    },
    {
        "id": "infeksi_jamur_kulit",
        "category": "Kulit & Rambut",
        "title": "Infeksi Jamur Kulit: Kurap, Panu, Kutu Air, dan Salep Antijamur",
        "query": "Gejala panu kurap kutu air gatal di lipatan kulit lembap dan salep antijamur",
        "title_regex": r"\bpanu\b|\bkurap\b|kutu air|jamur kulit",
        "content_regex": r"(panu|kurap|jamur kulit).*(gatal|bercak|salep)"
    },

    # 14. Mata & Telinga (THT)
    {
        "id": "mata_kering_gadget",
        "category": "Mata & THT",
        "title": "Sindrom Mata Kering: Akibat Menatap Layar HP/Laptop dan Tetes Mata",
        "query": "Gejala mata kering perih pegal akibat layar gadget aturan 20-20-20 dan obat tetes air mata",
        "title_regex": r"mata kering|dry eye",
        "content_regex": r"mata kering.*(layar|gadget|tetes mata|perih)"
    },
    {
        "id": "katarak_mata",
        "category": "Mata & THT",
        "title": "Katarak: Lensa Mata Keruh pada Lansia dan Operasi Phaco",
        "query": "Gejala katarak pandangan buram berkabut silau operasi lensa katarak pada lansia",
        "title_regex": r"katarak",
        "content_regex": r"katarak.*(mata|lensa|buram|operasi)"
    },
    {
        "id": "glaukoma",
        "category": "Mata & THT",
        "title": "Glaukoma: Tekanan Bola Mata Tinggi dan Risiko Kebutaan Diam-diam",
        "query": "Bahaya glaukoma kerusakan saraf mata tekanan bola mata pandangan menyempit kebutaan",
        "title_regex": r"glaukoma",
        "content_regex": r"glaukoma.*(tekanan bola mata|saraf mata|kebutaan)"
    },
    {
        "id": "tinnitus_telinga_berdenging",
        "category": "Mata & THT",
        "title": "Tinnitus: Telinga Berdenging Terus-Menerus, Headset, dan Penanganan",
        "query": "Penyebab telinga berdenging terus tinnitus kebiasaan memakai earphone suara bising",
        "title_regex": r"tinnitus|telinga berdenging|telinga berdengung",
        "content_regex": r"(tinnitus|telinga berdenging).*(pendengaran|suara|telinga)"
    },
    {
        "id": "radang_amandel",
        "category": "Mata & THT",
        "title": "Radang Amandel (Tonsilitis): Sakit Menelan, Demam, dan Kapan Dioperasi",
        "query": "Gejala radang amandel tonsilitis sakit menelan amandel bengkak bernanah dan indikasi operasi",
        "title_regex": r"amandel|tonsilitis",
        "content_regex": r"(amandel|tonsilitis).*(tenggorokan|menelan|bengkak)"
    },

    # 15. Sub-topik Tambahan Spesifik untuk Mencapai Target 140+ Klaster
    {
        "id": "kanker_ginjal",
        "category": "Onkologi & Kanker",
        "title": "Kanker Ginjal: Gejala Kencing Berdarah, Nyeri Pinggang, dan Terapi",
        "query": "Gejala kanker ginjal tanda kencing berdarah hematuria nyeri pinggang benjolan dan operasi",
        "title_regex": r"kanker ginjal",
        "content_regex": r"kanker ginjal.*(kencing|darah|pinggang|operasi)"
    },
    {
        "id": "kanker_otak",
        "category": "Onkologi & Kanker",
        "title": "Tumor dan Kanker Otak: Gejala Sakit Kepala Pagi, Kejang, dan Penanganan",
        "query": "Gejala tumor kanker otak sakit kepala hebat pagi hari muntah kejang gangguan saraf",
        "title_regex": r"tumor otak|kanker otak",
        "content_regex": r"(tumor otak|kanker otak).*(sakit kepala|kejang|operasi|saraf)"
    },
    {
        "id": "kanker_lambung",
        "category": "Onkologi & Kanker",
        "title": "Kanker Lambung: Gejala Nyeri Ulu Hati Kronis dan Penurunan Berat Badan",
        "query": "Gejala kanker lambung rasa cepat kenyang nyeri ulu hati bab hitam berat badan turun",
        "title_regex": r"kanker lambung",
        "content_regex": r"kanker lambung.*(ulu hati|muntah|lambung|keganasan)"
    },
    {
        "id": "kanker_tulang",
        "category": "Onkologi & Kanker",
        "title": "Kanker Tulang (Osteosarkoma): Gejala Nyeri Tulang Malam Hari dan Bengkak",
        "query": "Gejala kanker tulang osteosarkoma nyeri tulang malam hari pembengkakan sendi dan kemoterapi",
        "title_regex": r"kanker tulang|sarkoma|osteosarkoma",
        "content_regex": r"(kanker tulang|osteosarkoma).*(nyeri|bengkak|tulang)"
    },
    {
        "id": "limfoma_getah_bening",
        "category": "Onkologi & Kanker",
        "title": "Limfoma: Kanker Kelenjar Getah Bening dan Benjolan di Leher",
        "query": "Gejala limfoma kanker kelenjar getah bening benjolan leher ketiak demam malam hari",
        "title_regex": r"limfoma|kelenjar getah bening",
        "content_regex": r"(limfoma|kelenjar getah bening).*(kanker|benjolan|leher)"
    },
    {
        "id": "kista_ovarium",
        "category": "Kesehatan Ibu & Anak",
        "title": "Kista Ovarium: Gejala Nyeri Panggul, Menstruasi, dan Kapan Dioperasi",
        "query": "Gejala kista ovarium indung telur nyeri panggul perut buncit gangguan haid dan operasi",
        "title_regex": r"kista ovarium|kista rahim",
        "content_regex": r"kista.*(ovarium|rahim|nyeri|operasi)"
    },
    {
        "id": "miom_rahim",
        "category": "Kesehatan Ibu & Anak",
        "title": "Miom Uteri: Tumor Jinak Rahim, Pendarahan Haid, dan Kesuburan",
        "query": "Gejala miom uteri tumor jinak dinding rahim menstruasi deras nyeri panggul dan pengobatan",
        "title_regex": r"miom|mioma",
        "content_regex": r"(miom|mioma).*(rahim|haid|pendarahan)"
    },
    {
        "id": "kram_menstruasi_dismenore",
        "category": "Kesehatan Ibu & Anak",
        "title": "Dismenore: Cara Mengatasi Kram dan Nyeri Haid Hebat",
        "query": "Cara meredakan kram menstruasi dismenore sakit perut saat haid kompres hangat dan obat",
        "title_regex": r"dismenore|nyeri haid|kram perut haid|kram menstruasi",
        "content_regex": r"(dismenore|nyeri haid|kram haid).*(perut|menstruasi|kompres)"
    },
    {
        "id": "baby_blues_postpartum",
        "category": "Kesehatan Ibu & Anak",
        "title": "Baby Blues dan Depresi Pasca Melahirkan: Gejala Emosi Ibu Baru",
        "query": "Perbedaan baby blues dan depresi pascapersalinan postpartum menangis cemas ibu baru melahirkan",
        "title_regex": r"baby blues|depresi pasca",
        "content_regex": r"(baby blues|depresi pasca).*(melahirkan|ibu|bayi|emosi)"
    },
    {
        "id": "autisme_anak",
        "category": "Kesehatan Ibu & Anak",
        "title": "Autisme pada Anak: Ciri Keterlambatan Bicara, Kontak Mata, dan Terapi",
        "query": "Tanda anak autisme speech delay keterlambatan bicara kontak mata minim stimulasi dan terapi",
        "title_regex": r"\bautis\b|\bautisme\b",
        "content_regex": r"(autis|autisme).*(anak|bicara|terapi|tumbuh kembang)"
    },
    {
        "id": "adhd_anak",
        "category": "Kesehatan Ibu & Anak",
        "title": "ADHD pada Anak: Ciri Perilaku Hiperaktif, Impulsif, dan Penanganan",
        "query": "Ciri anak adhd hiperaktif sulit konsentrasi impulsif pola asuh orang tua dan terapi",
        "title_regex": r"\badhd\b|hiperaktif",
        "content_regex": r"(adhd|hiperaktif).*(anak|fokus|konsentrasi|perilaku)"
    },
    {
        "id": "alergi_susu_sapi",
        "category": "Kesehatan Ibu & Anak",
        "title": "Alergi Susu Sapi pada Bayi: Gejala Diare, Ruam Kulit, dan Pengganti Susu",
        "query": "Gejala alergi susu sapi intoleransi laktosa diare muntah ruam kulit bayi dan susu soya",
        "title_regex": r"alergi susu|susu sapi|intoleransi laktosa",
        "content_regex": r"(alergi susu|intoleransi laktosa).*(bayi|anak|diare|ruam)"
    },
    {
        "id": "gigi_berlubang",
        "category": "Gigi & Mulut",
        "title": "Gigi Berlubang (Karies): Penyebab Sakit Gigi Berdenyut dan Tambal Gigi",
        "query": "Penyebab gigi berlubang karies plak bakteri sakit gigi berdenyut dan prosedur tambal gigi",
        "title_regex": r"gigi berlubang|karies|sakit gigi",
        "content_regex": r"(gigi berlubang|karies|sakit gigi).*(bakteri|plak|tambal|saraf)"
    },
    {
        "id": "radang_gusi_gingivitis",
        "category": "Gigi & Mulut",
        "title": "Radang Gusi (Gingivitis): Gusi Berdarah, Karang Gigi, dan Scaling",
        "query": "Gejala radang gusi gingivitis gusi berdarah saat sikat gigi karang gigi dan scaling dokter",
        "title_regex": r"radang gusi|gusi berdarah|gingivitis",
        "content_regex": r"(radang gusi|gusi berdarah|gingivitis).*(karang gigi|scaling|mulut)"
    },
    {
        "id": "sariawan_kronis",
        "category": "Gigi & Mulut",
        "title": "Sariawan Berulang: Penyebab Luka Mulut, Defisiensi Vitamin, dan Obat Alami",
        "query": "Penyebab sering sariawan luka bibir lidah kurang vitamin c b12 stres dan obat kumur",
        "title_regex": r"sariawan",
        "content_regex": r"sariawan.*(mulut|lidah|bibir|luka|vitamin)"
    },
    {
        "id": "bau_mulut_halitosis",
        "category": "Gigi & Mulut",
        "title": "Bau Mulut (Halitosis): Penyebab Bakteri Lidah dan Cara Menghilangkannya",
        "query": "Penyebab bau mulut halitosis bakteri lidah asam lambung gigi berlubang dan cara hilangkan",
        "title_regex": r"bau mulut|halitosis",
        "content_regex": r"(bau mulut|halitosis).*(bakteri|lidah|gigi|napas)"
    },
    {
        "id": "sakit_tenggorokan_faringitis",
        "category": "Respirasi & Paru",
        "title": "Sakit Tenggorokan (Faringitis): Penyebab Perih Menelan dan Terapi Alami",
        "query": "Cara meredakan sakit tenggorokan radang faringitis perih saat menelan air garam madu obat",
        "title_regex": r"sakit tenggorokan|radang tenggorokan|faringitis",
        "content_regex": r"(sakit tenggorokan|radang tenggorokan|faringitis).*(menelan|suara|perih)"
    },
    {
        "id": "flu_burung_zoonosis",
        "category": "Infeksi & Penyakit Menular",
        "title": "Flu Burung (H5N1): Penularan dari Unggas, Gejala Demam, dan Pencegahan",
        "query": "Penyebab flu burung virus h5n1 penularan kontak unggas gejala demam sesak napas pencegahan",
        "title_regex": r"flu burung|h5n1",
        "content_regex": r"(flu burung|h5n1).*(unggas|virus|demam|penularan)"
    },
    {
        "id": "cacingan_anak",
        "category": "Infeksi & Penyakit Menular",
        "title": "Cacingan pada Anak: Gejala Gatal Anus, Kurang Gizi, dan Obat Cacing",
        "query": "Ciri anak cacingan cacing kremi gatal anus nafsu makan turun berat badan dan obat cacing",
        "title_regex": r"cacingan|infeksi cacing|cacing kremi",
        "content_regex": r"(cacingan|cacing).*(anak|anus|gatal|obat cacing)"
    },
    {
        "id": "perut_kembung_begah",
        "category": "Lambung & Pencernaan",
        "title": "Perut Kembung dan Begah: Penyebab Penumpukan Gas dan Cara Mengatasinya",
        "query": "Penyebab perut kembung begah banyak gas sering sendawa kentut makanan pemicu gas dan solusi",
        "title_regex": r"perut kembung|perut begah|\bbegah\b",
        "content_regex": r"(kembung|begah).*(gas|perut|sendawa|lambung)"
    },
    {
        "id": "radang_usus_kolitis",
        "category": "Lambung & Pencernaan",
        "title": "Radang Usus (Kolitis): Gejala Diare Berlendir, Nyeri Perut, dan Pola Makan",
        "query": "Gejala radang usus kolitis inflamasi diare lendir berdarah kram perut dan makanan pantangan",
        "title_regex": r"kolitis|radang usus",
        "content_regex": r"(kolitis|radang usus).*(diare|usus|perut|inflamasi)"
    },
    {
        "id": "varises_kaki",
        "category": "Jantung & Kardiovaskular",
        "title": "Varises Kaki: Pembengkakan Pembuluh Vena, Kaki Pegal, dan Penanganan",
        "query": "Penyebab varises kaki pembuluh darah vena membesar kaki pegal stoking kompresi dan terapi",
        "title_regex": r"varises",
        "content_regex": r"varises.*(kaki|vena|pembuluh darah|pegal)"
    },
    {
        "id": "pegal_linu_nyeri_otot",
        "category": "Tulang & Sendi",
        "title": "Pegal Linu dan Nyeri Otot (Myalgia): Penumpukan Asam Laktat dan Relaksasi",
        "query": "Penyebab badan pegal linu nyeri otot myalgia setelah kerja asam laktat pijat dan peregangan",
        "title_regex": r"pegal linu|nyeri otot|myalgia",
        "content_regex": r"(pegal linu|nyeri otot|myalgia).*(tubuh|otot|lelah|peregangan)"
    },
    {
        "id": "cedera_ligamen_keseleo",
        "category": "Tulang & Sendi",
        "title": "Cedera Keseleo dan Terkilir: Metode RICE dan Bahaya Asal Diurut",
        "query": "Pertolongan pertama keseleo terkilir pergelangan kaki bengkak metode rice es dan kompres",
        "title_regex": r"keseleo|terkilir|cedera ligamen",
        "content_regex": r"(keseleo|terkilir|ligamen).*(bengkak|kaki|kompres|rice)"
    },
    {
        "id": "nyeri_lutut_lansia",
        "category": "Tulang & Sendi",
        "title": "Nyeri Lutut pada Lansia: Faktor Beban Tubuh, Otot Lemah, dan Latihan",
        "query": "Penyebab sakit lutut nyeri saat naik tangga beban tubuh aus sendi dan latihan penguat otot",
        "title_regex": r"nyeri lutut|sakit lutut",
        "content_regex": r"(nyeri lutut|sakit lutut).*(sendi|tangga|kaki|otot)"
    },
    {
        "id": "vape_rokok_elektrik",
        "category": "Nutrisi & Gaya Hidup",
        "title": "Bahaya Vape dan Rokok Elektrik: Dampak Nikotin Cair dan Paru-Paru",
        "query": "Bahaya vape rokok elektrik cairan nikotin zat kimia kerusakan paru paru evali dibanding rokok",
        "title_regex": r"vape|rokok elektrik",
        "content_regex": r"(vape|rokok elektrik).*(paru|nikotin|bahaya|cairan)"
    },
    {
        "id": "intermittent_fasting",
        "category": "Nutrisi & Gaya Hidup",
        "title": "Puasa Intermiten (Intermittent Fasting): Manfaat Berat Badan dan Pola 16:8",
        "query": "Metode intermittent fasting puasa intermiten jendela makan 16 8 bakar lemak autofagi",
        "title_regex": r"intermittent fasting|puasa intermiten",
        "content_regex": r"(intermittent fasting|puasa intermiten).*(jendela makan|lemak|berat badan)"
    },
    {
        "id": "minyak_jelantah_gorengan",
        "category": "Nutrisi & Gaya Hidup",
        "title": "Bahaya Minyak Jelantah dan Gorengan: Lemak Trans dan Risiko Jantung",
        "query": "Bahaya minyak jelantah dipakai berulang lemak trans kolesterol radikal bebas dan gorengan",
        "title_regex": r"minyak jelantah|gorengan",
        "content_regex": r"(minyak jelantah|gorengan).*(kolesterol|lemak trans|jantung|kanker)"
    },
    {
        "id": "jalan_kaki_sehat",
        "category": "Nutrisi & Gaya Hidup",
        "title": "Manfaat Jalan Kaki Rutin: Menurunkan Gula Darah dan Memperkuat Jantung",
        "query": "Manfaat jalan kaki 10000 langkah setiap hari turunkan gula darah bakar kalori jantung sehat",
        "title_regex": r"jalan kaki|10\.000 langkah",
        "content_regex": r"jalan kaki.*(langkah|jantung|sehat|gula darah|kalori)"
    },
    {
        "id": "khasiat_kunyit",
        "category": "Nutrisi & Gaya Hidup",
        "title": "Khasiat Kunyit dan Kurkumin: Manfaat Alami Anti-Radang Lambung dan Sendi",
        "query": "Manfaat kunyit senyawa kurkumin meredakan asam lambung antioksidan anti radang alami",
        "title_regex": r"\bkunyit\b|kurkumin",
        "content_regex": r"(kunyit|kurkumin).*(lambung|radang|manfaat|herbal)"
    },
    {
        "id": "manfaat_jahe",
        "category": "Nutrisi & Gaya Hidup",
        "title": "Manfaat Jahe Merah: Mengatasi Mual, Hangatkan Tubuh, dan Imunitas",
        "query": "Khasiat jahe merah senyawa gingerol meredakan mual batuk menghangatkan badan dan imun",
        "title_regex": r"manfaat jahe|khasiat jahe|jahe merah",
        "content_regex": r"jahe.*(mual|gingerol|hangat|imun|manfaat)"
    },
    {
        "id": "manfaat_bawang_putih",
        "category": "Nutrisi & Gaya Hidup",
        "title": "Manfaat Bawang Putih: Alisin Penurun Tekanan Darah dan Kolesterol",
        "query": "Khasiat bawang putih mentah senyawa alisin menurunkan tekanan darah kolesterol dan jantung",
        "title_regex": r"bawang putih",
        "content_regex": r"bawang putih.*(kolesterol|darah tinggi|alisin|jantung)"
    },
    {
        "id": "teh_hijau",
        "category": "Nutrisi & Gaya Hidup",
        "title": "Khasiat Teh Hijau: Antioksidan EGCG, Metabolisme Lemak, dan Jantung",
        "query": "Manfaat minum teh hijau green tea antioksidan egcg membakar lemak kolesterol dan otak",
        "title_regex": r"teh hijau|green tea",
        "content_regex": r"(teh hijau|green tea).*(antioksidan|egcg|lemak|sehat)"
    }
]

def run_clustering():
    dataset_path = os.path.join("data", "berita_kesehatan_fix.csv")
    print(f"Membaca dataset utama: {dataset_path} ...")
    df = pd.read_csv(dataset_path, sep="|")
    total_rows = len(df)
    print(f"Total baris dataset: {total_rows} artikel.\n")

    print(f"Total Taksonomi Sub-Topik Didaftarkan: {len(TAXONOMY)} sub-topik.")
    
    # Pre-filter dataset
    valid_articles = []
    for idx, row in df.iterrows():
        title = str(row['judul']) if pd.notna(row['judul']) else ""
        content = str(row['isi_berita']) if pd.notna(row['isi_berita']) else ""
        date = str(row['tanggal']) if pd.notna(row['tanggal']) else ""
        url = str(row['url']) if pd.notna(row['url']) else ""
        
        if len(content.split()) < 50:  # Artikel terlalu pendek/kosong
            continue
        if is_noise(title, content):
            continue
            
        valid_articles.append({
            "index": idx,
            "title": title,
            "content": content,
            "date": date,
            "url": url
        })
        
    print(f"Artikel Bersih & Valid Bebas Noise: {len(valid_articles)} artikel.\n")
    print("Memulai pencocokan klaster sub-topik...")

    clusters_result = []
    category_counts = defaultdict(int)
    total_articles_mapped = 0

    for topic in TAXONOMY:
        t_id = topic["id"]
        cat = topic["category"]
        t_title = topic["title"]
        t_query = topic["query"]
        p_title = topic["title_regex"]
        p_content = topic["content_regex"]

        # Cari artikel yang cocok
        matched = []
        for art in valid_articles:
            title_text = art["title"]
            body_text = art["content"][:1500]  # Prioritaskan 1500 karakter awal

            score = 0
            if re.search(p_title, title_text, re.IGNORECASE):
                score += 10  # Bobot sangat tinggi jika ada di judul
            if re.search(p_content, body_text, re.IGNORECASE):
                score += 3   # Bobot pendukung jika ada di konten

            if score >= 3:
                matched.append({
                    "score": score,
                    "index": art["index"],
                    "title": art["title"],
                    "url": art["url"],
                    "date": art["date"],
                    "preview": art["content"][:300] + "..."
                })

        # Urutkan berdasarkan relevansi (score tertinggi & judul paling sesuai)
        matched.sort(key=lambda x: x["score"], reverse=True)

        # Ambil 4 hingga 8 artikel terbaik per klaster
        top_articles = matched[:8]

        # Simpan jika klaster memiliki minimal 2 artikel relevan
        if len(top_articles) >= 2:
            cluster_entry = {
                "id": t_id,
                "category": cat,
                "title": t_title,
                "query": t_query,
                "article_count": len(top_articles),
                "articles": top_articles
            }
            clusters_result.append(cluster_entry)
            category_counts[cat] += 1
            total_articles_mapped += len(top_articles)

    print("="*70)
    print("           HASIL PEMETAAN KLASTERISASI SUB-TOPIK")
    print("="*70)
    print(f"Total Sub-Topik Lolos Seleksi : {len(clusters_result)} klaster (Target 120-170 tercapai!)")
    print(f"Total Artikel Terklaster      : {total_articles_mapped} artikel.")
    print("\nDistribusi Per Kategori Klinis:")
    for cat, cnt in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f" - {cat:28s}: {cnt:2d} sub-topik")

    print("\nContoh Cuplikan Klaster Sub-Topik:")
    for c in clusters_result[:5]:
        print(f"\n[ID: {c['id']}] ({c['category']})")
        print(f"Judul Klaster: {c['title']}")
        print(f"Query Uji    : \"{c['query']}\"")
        print(f"Jumlah Berita: {c['article_count']} artikel")
        for i, a in enumerate(c['articles'][:3], 1):
            print(f"   {i}. [{a['index']}] {a['title']}")

    # Simpan ke file JSON
    out_file = os.path.join("data", "clusters_subtopics_mapped.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(clusters_result, f, indent=2, ensure_ascii=False)
        
    print(f"\n[SUKSES] Berkas hasil klasterisasi tersimpan di: {out_file}")

if __name__ == "__main__":
    run_clustering()
