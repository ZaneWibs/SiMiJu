"""
main.py — FastAPI + Uvicorn v3
Sistem Deteksi Kemiripan Judul Skripsi
STAB Negeri Raden Wijaya Wonogiri

Fitur:
  - Normalisasi sinonim domain pendidikan Buddha
  - Panel admin upload CSV (password protected)
  - API log pencarian ke SQLite
  - Pencarian judul berdasarkan lokasi (opsional, dari kolom Tempat CSV)

Jalankan:
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000
"""

import re, math, io, sqlite3, secrets, os
from datetime import datetime
from pathlib import Path
from collections import Counter
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Header, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ══════════════════════════════════════════════════════════════
#  KONFIGURASI
# ══════════════════════════════════════════════════════════════
BASE_DIR           = Path(__file__).parent
DATASET_PATH       = BASE_DIR / "skripsi_dataset.csv"
DATA_DIR           = Path(os.getenv("DATA_DIR", str(BASE_DIR)))
DB_PATH            = DATA_DIR / "log.db"

KEYWORD_WEIGHT     = 3
THR_SANGAT_MIRIP   = 0.68
THR_TAMPIL         = 0.18
THR_TOPIK_JENUH_KW = 0.35
THR_TOPIK_JENUH_N  = 20

W_TFIDF_WORD = 0.10
W_TFIDF_CHAR = 0.05
W_BM25       = 0.15
W_JACCARD_KW = 0.70   # kata kunci exact match jadi penentu utama

# Password admin — set via environment variable ADMIN_PASSWORD di Render
ADMIN_PASSWORD      = os.getenv("ADMIN_PASSWORD", "stab2025")
# Token disimpan di memory (reset saat server restart)
_valid_tokens: set[str] = set()

# ══════════════════════════════════════════════════════════════
#  KAMUS SINONIM — domain pendidikan Buddha (Prioritas #3)
#  Format: kata_target -> [sinonim, sinonim, ...]
#  Semua sinonim akan dinormalisasi ke kata_target saat tokenisasi
# ══════════════════════════════════════════════════════════════
SINONIM: dict[str, list[str]] = {
    # Peserta didik
    "siswa"          : ["peserta didik", "murid", "anak didik", "pelajar"],
    "mahasiswa"      : ["peserta didik perguruan tinggi", "mahasiswi"],
    # Pendidik
    "guru"           : ["pengajar", "pendidik", "tenaga pengajar", "fasilitator"],
    "dosen"          : ["pengajar perguruan tinggi", "tenaga pendidik"],
    # Proses belajar
    "pembelajaran"   : ["belajar mengajar", "kegiatan belajar", "proses belajar",
                        "proses pembelajaran", "kegiatan pembelajaran"],
    "mengajar"       : ["pengajaran", "penyampaian materi"],
    "motivasi"       : ["semangat belajar", "dorongan belajar", "minat belajar"],
    "prestasi"       : ["hasil belajar", "capaian belajar", "nilai", "raport"],
    "kompetensi"     : ["kemampuan", "kecakapan", "keahlian", "keterampilan"],
    # Media & metode
    "media pembelajaran" : ["alat pembelajaran", "bahan ajar", "media ajar",
                             "sarana pembelajaran", "media pendidikan"],
    "metode"         : ["teknik", "cara", "pendekatan", "strategi"],
    "model"          : ["pola", "desain pembelajaran"],
    "aplikasi"       : ["app", "perangkat lunak", "software", "platform"],
    "digital"        : ["online", "daring", "berbasis teknologi", "elektronik"],
    "video"          : ["film", "rekaman", "audio visual", "audiovisual"],
    # Agama Buddha
    "buddha"         : ["buddhis", "buddhisme", "budhis"],
    "dhamma"         : ["dharma", "ajaran buddha", "ajaran buddhis"],
    "vihara"         : ["wihara", "cetiya", "tempat ibadah buddha"],
    "tripitaka"      : ["tipitaka", "kitab suci buddha"],
    "waisak"         : ["vesak", "hari raya waisak", "hari waisak"],
    "meditasi"       : ["samadhi", "vipassana", "meditasi buddhis"],
    "sila"           : ["moralitas", "etika buddhis", "pancha sila"],
    "sangha"         : ["komunitas buddhis", "umat buddha"],
    "jataka"         : ["cerita jataka", "kisah jataka"],
    # Evaluasi
    "evaluasi"       : ["penilaian", "asesmen", "assessment", "pengukuran"],
    "ujian"          : ["tes", "test", "ulangan", "kuis", "quiz"],
    # Karakter & perilaku
    "karakter"       : ["kepribadian", "watak", "budi pekerti", "akhlak"],
    "komunikasi"     : ["interaksi", "dialog", "percakapan"],
    # Kurikulum
    "kurikulum"      : ["silabus", "rencana pembelajaran", "rpp"],
    "merdeka belajar": ["kurikulum merdeka", "kurikulum prototipe"],
    # Lain-lain
    "literasi"       : ["kemampuan membaca", "minat baca", "budaya baca"],
    "moderasi"       : ["toleransi", "kerukunan", "harmoni"],
    "efektivitas"    : ["keefektifan", "efektif"],
    "pengembangan"   : ["pengembangan", "development", "pembuatan"],
}

# Buat lookup terbalik: sinonim → target
_SINONIM_LOOKUP: dict[str, str] = {}
for target, synonyms in SINONIM.items():
    for syn in synonyms:
        _SINONIM_LOOKUP[syn.lower()] = target

def normalize_synonyms(text: str) -> str:
    """Ganti sinonim dalam teks dengan kata target (frasa dulu, lalu kata)."""
    text_lower = text.lower()
    # Frasa dulu (urut panjang → pendek) agar "media pembelajaran" tidak dipecah
    for syn in sorted(_SINONIM_LOOKUP, key=len, reverse=True):
        if syn in text_lower:
            text_lower = text_lower.replace(syn, _SINONIM_LOOKUP[syn])
    return text_lower

# ══════════════════════════════════════════════════════════════
#  STOPWORDS
# ══════════════════════════════════════════════════════════════
STOPWORDS = {
    'dan','atau','yang','di','ke','dari','pada','dalam','untuk','dengan',
    'adalah','ini','itu','oleh','juga','sudah','serta','terhadap','sebagai',
    'melalui','antara','secara','bagi','akan','tidak','dapat','ada','atas',
    'bukan','bila','saat','suatu','sebuah','setiap','lebih','namun','tentang',
    'mengenai','apabila','kepada','bahwa','karena','ketika','setelah','sebelum',
    'sejak','hingga','sampai','maka','agar','supaya','jika','pun','lah','kah',
    'nya','si','sang','para','berbagai','beberapa','banyak','sedikit',
    'sangat','cukup','hanya','saja','pula','tetapi','melainkan','padahal',
    'meskipun','walaupun','sehingga',
    'studi','analisis','pengaruh','pengembangan','implementasi','evaluasi',
    'hubungan','peran','faktor','tingkat','upaya','kajian','penelitian',
    'efektivitas','efisiensi','kualitas','gambaran','deskripsi','tinjauan',
    'survei','perbandingan','identifikasi','penerapan','pemanfaatan',
    'kabupaten','kecamatan','desa','kota','provinsi','negeri','swasta',
}

# ══════════════════════════════════════════════════════════════
#  BM25 manual
# ══════════════════════════════════════════════════════════════
class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1 = k1; self.b = b
        self.corpus = corpus
        self.N      = len(corpus)
        self.avgdl  = sum(len(d) for d in corpus) / max(self.N, 1)
        df = Counter()
        for doc in corpus:
            for term in set(doc):
                df[term] += 1
        self.idf = {
            t: math.log((self.N - f + 0.5) / (f + 0.5) + 1)
            for t, f in df.items()
        }

    def get_scores(self, query):
        scores = np.zeros(self.N)
        for term in query:
            if term not in self.idf:
                continue
            idf = self.idf[term]
            for i, doc in enumerate(self.corpus):
                tf = doc.count(term)
                dl = len(doc)
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[i] += idf * (tf * (self.k1 + 1)) / denom
        return scores

# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════
def tokenize(text: str, remove_stop: bool = True) -> list[str]:
    text   = normalize_synonyms(text)          # ← sinonim dinormalisasi dulu
    text   = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = text.split()
    if remove_stop:
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    return tokens

def parse_keywords(kw_str: str) -> list[str]:
    kw_str = re.sub(r'[;|/]', ',', str(kw_str))
    return [k.strip().lower() for k in kw_str.split(',') if k.strip()]



def build_weighted_text(judul: str, kata_kunci: str, weight: int = KEYWORD_WEIGHT) -> str:
    return ' '.join(tokenize(judul) + tokenize(kata_kunci) * weight)



def phrase_specificity(phrase_norm: str) -> float:
    """
    Spesifisitas sebuah frasa kata kunci: nilai [0.20, 1.00]

    Makin tinggi = makin spesifik = makin besar kontribusinya dalam similarity.
    Makin rendah = makin umum/ambigu = makin kecil kontribusinya.

    Gabungan dua komponen:
    ┌─────────────────────────────────────────────────────────────────┐
    │  PANJANG FRASA  (bobot 40%)                                     │
    │  1 kata  → 0.35  │  2 kata → 0.70  │  ≥3 kata → 1.00          │
    ├─────────────────────────────────────────────────────────────────┤
    │  IDF DI DATASET  (bobot 60%)                                    │
    │  Frasa dikenal   → pakai IDF frasa itu langsung                 │
    │  Kata tunggal    → pakai frekuensi kata tsb di seluruh frasa   │
    │    ("minat" → muncul di banyak frasa → IDF rendah → umum)      │
    │  Tidak dikenal   → anggap spesifik (IDF tinggi)                 │
    └─────────────────────────────────────────────────────────────────┘

    Contoh efek nyata:
      "minat"              → panjang rendah + IDF rendah  ≈ 0.35–0.42
      "minat belajar"      → panjang sedang + IDF sedang  ≈ 0.60–0.70
      "minat berkunjung"   → panjang sedang + IDF sedang  ≈ 0.60–0.70
      "meditasi vipassana" → panjang sedang + IDF tinggi  ≈ 0.75–0.85
      "tripitaka"          → panjang rendah + IDF tinggi  ≈ 0.65–0.75
    """
    n_words      = len(phrase_norm.split())
    length_score = min(1.0, n_words * 0.35)   # 1→0.35, 2→0.70, 3+→1.00

    if hasattr(state, 'kw_idf') and state.kw_idf_max > 0:
        idf_max = state.kw_idf_max

        if phrase_norm in state.kw_idf:
            # Frasa dikenal persis di dataset
            idf_norm = state.kw_idf[phrase_norm] / idf_max

        elif n_words == 1 and phrase_norm in state.word_in_kw_freq:
            # Kata tunggal: pakai frekuensi kemunculan kata ini di berbagai frasa
            # Contoh: "minat" ada di "minat belajar", "minat berkunjung", dst.
            #  → frekuensi tinggi → IDF rendah → spesifisitas rendah
            freq     = state.word_in_kw_freq[phrase_norm]
            idf_raw  = math.log((state.total_docs + 1) / (freq + 1)) + 1.0
            idf_norm = idf_raw / idf_max

        else:
            # Tidak dikenal → asumsikan spesifik (nilai tengah-tinggi)
            idf_norm = 0.80

    else:
        idf_norm = 0.50   # fallback sebelum load_model selesai

    return max(0.20, 0.40 * length_score + 0.60 * idf_norm)


def jaccard_kw(list_a: list[str], list_b: list[str]) -> tuple[float, float]:
    """
    Weighted Jaccard similarity antar dua list kata kunci.
    Mengembalikan tuple: (jaccard_score, match_scale)

    ── Perbedaan dari versi lama ─────────────────────────────────────
    Versi lama: semua frasa dianggap setara (bobot = 1.0 per frasa).
    Versi baru: tiap frasa diberi BOBOT sesuai phrase_specificity():
      • Kata umum/pendek → bobot kecil (tidak bisa mendominasi skor)
      • Frasa spesifik   → bobot besar (lebih bermakna jika cocok)

    ── Cara kerja ────────────────────────────────────────────────────
    Exact match  : kontribusi = specificity(frasa)          (penuh)
    Partial match: kontribusi = 0.5 × specificity(frasa_pendek)
      → Frasa pendek generik seperti "minat" mendapat penalti ganda:
        (1) faktor 0.5 dari partial match
        (2) specificity("minat") yang sudah rendah
      → Hasilnya: skor partial "minat" ∈ "minat belajar" sangat kecil

    Weighted Jaccard = Σ kontribusi intersection / Σ bobot union

    match_scale : pengali [0–1] berdasarkan BERAPA BANYAK kw input cocok
      (mekanisme balancing dari sesi sebelumnya, tetap dipertahankan)
    """
    if not list_a or not list_b:
        return 0.0, 0.0

    def norm(phrase: str) -> str:
        return re.sub(r'\s+', ' ', phrase.strip().lower())

    set_a = {norm(k) for k in list_a}
    set_b = {norm(k) for k in list_b}

    # ── Exact match ────────────────────────────────────────────
    exact = set_a & set_b

    # ── Partial match: satu frasa mengandung frasa lain ────────
    partial_matched_a: set[str] = set()
    partial_weight: float = 0.0
    for a in set_a - exact:
        for b in set_b - exact:
            if a in b:
                # a lebih pendek/generik → specificity(a) menentukan kontribusi
                # Contoh: "minat" dalam "minat belajar"
                #   → spec("minat") ≈ 0.38 → kontribusi = 0.5 × 0.38 = 0.19
                partial_weight += 0.5 * phrase_specificity(a)
                partial_matched_a.add(a)
                break
            elif b in a:
                # b lebih pendek/generik
                partial_weight += 0.5 * phrase_specificity(b)
                partial_matched_a.add(a)
                break

    # ── Weighted Jaccard ───────────────────────────────────────
    union_set    = set_a | set_b
    union_weight = sum(phrase_specificity(p) for p in union_set)
    exact_weight = sum(phrase_specificity(p) for p in exact)

    if union_weight == 0:
        return 0.0, 0.0

    score = min((exact_weight + partial_weight) / union_weight, 1.0)

    # ── match_scale: balancing jumlah kw yang cocok ────────────
    n_input   = len(set_a)
    n_matched = len(exact) + len(partial_matched_a)

    if n_input == 0:
        match_scale = 0.0
    elif n_input == 1:
        match_scale = 0.40 if n_matched >= 1 else 0.0
    else:
        match_ratio = n_matched / n_input
        if match_ratio <= 0.25:
            match_scale = 0.30
        elif match_ratio <= 0.50:
            match_scale = 0.60
        elif match_ratio <= 0.75:
            match_scale = 0.85
        else:
            match_scale = 1.00

    return score, match_scale

# ══════════════════════════════════════════════════════════════
#  DATABASE LOG (SQLite) — Prioritas #4 (riwayat pencarian)
# ══════════════════════════════════════════════════════════════
def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS log_pencarian (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            waktu      TEXT NOT NULL,
            judul      TEXT NOT NULL,
            kata_kunci TEXT NOT NULL,
            status     TEXT NOT NULL,
            n_mirip    INTEGER,
            avg_sat    REAL
        )
    """)
    con.commit()
    con.close()

def log_pencarian(judul: str, kata_kunci: str, status: str, n_mirip: int, avg_sat: float):
    try:
        con = sqlite3.connect(DB_PATH)
        con.execute(
            "INSERT INTO log_pencarian (waktu, judul, kata_kunci, status, n_mirip, avg_sat) VALUES (?,?,?,?,?,?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), judul, kata_kunci, status, n_mirip, avg_sat)
        )
        con.commit()
        con.close()
    except Exception:
        pass  # log tidak boleh crash aplikasi utama

def get_log(limit: int = 100) -> list[dict]:
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT id, waktu, judul, kata_kunci, status, n_mirip, avg_sat "
        "FROM log_pencarian ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    con.close()
    return [
        {"id": r[0], "waktu": r[1], "judul": r[2], "kata_kunci": r[3],
         "status": r[4], "n_mirip": r[5], "avg_sat": r[6]}
        for r in rows
    ]

# ══════════════════════════════════════════════════════════════
#  MODEL STATE
# ══════════════════════════════════════════════════════════════
class ModelState:
    pass

state = ModelState()

def load_model():
    print("🔄 Memuat dataset dan membangun model ML...")
    df = pd.read_csv(DATASET_PATH, encoding='utf-8-sig')

    # Normalkan nama kolom — CSV bisa punya 2 atau 3 kolom (Judul, Kata_Kunci, Tempat)
    cols = list(df.columns)
    if len(cols) >= 3:
        df.columns = ['Judul', 'Kata_Kunci', 'Tempat'] + cols[3:]
        df = df[['Judul', 'Kata_Kunci', 'Tempat']]
    else:
        df.columns = ['Judul', 'Kata_Kunci'] + cols[2:]
        df = df[['Judul', 'Kata_Kunci']]
        df['Tempat'] = ''

    df = df.dropna(subset=['Judul']).reset_index(drop=True)
    df['Judul']      = df['Judul'].astype(str).str.strip()
    df['Kata_Kunci'] = df['Kata_Kunci'].fillna('').astype(str).str.strip()
    df['Tempat']     = df['Tempat'].fillna('').astype(str).str.strip()
    df['Weighted_Text'] = df.apply(
        lambda r: build_weighted_text(r['Judul'], r['Kata_Kunci']), axis=1
    )
    df['KW_List'] = df['Kata_Kunci'].apply(parse_keywords)

    n_tempat = (df['Tempat'] != '').sum()
    print(f"   📍 {n_tempat} judul memiliki data lokasi dari {len(df)} total")

    tfidf_word = TfidfVectorizer(ngram_range=(1, 3), min_df=1, max_df=0.90,
                                  sublinear_tf=True, analyzer='word')
    mat_word   = tfidf_word.fit_transform(df['Weighted_Text'])

    tfidf_char = TfidfVectorizer(ngram_range=(3, 5), min_df=1, max_df=0.90,
                                  sublinear_tf=True, analyzer='char_wb')
    mat_char   = tfidf_char.fit_transform(df['Weighted_Text'])

    corpus_tokens = df['Weighted_Text'].apply(str.split).tolist()
    bm25          = BM25(corpus_tokens, k1=1.5, b=0.75)
    all_kw        = [kw for kwlist in df['KW_List'] for kw in kwlist]

    # ── Bangun IDF frasa & kata untuk keyword specificity weighting ──────
    # Digunakan oleh phrase_specificity() untuk menentukan seberapa "umum"
    # suatu kata kunci — kata umum seperti "minat" mendapat bobot lebih kecil.
    def _norm_kw(k: str) -> str:
        return re.sub(r'\s+', ' ', k.strip().lower())

    # IDF tingkat frasa: seberapa jarang frasa persis ini muncul di dataset
    kw_doc_freq_map: dict[str, int] = {}
    # IDF tingkat kata: seberapa jarang kata ini muncul di BERBAGAI frasa
    # (misal "minat" muncul di "minat belajar", "minat berkunjung", dst.)
    word_in_kw_freq: Counter = Counter()

    for kw_list in df['KW_List']:
        phrase_set_doc: set[str] = set()
        word_set_doc:   set[str] = set()
        for kw in kw_list:
            p = _norm_kw(kw)
            phrase_set_doc.add(p)
            word_set_doc.update(p.split())
        for p in phrase_set_doc:
            kw_doc_freq_map[p] = kw_doc_freq_map.get(p, 0) + 1
        for w in word_set_doc:
            word_in_kw_freq[w] += 1

    N = len(df)
    kw_idf: dict[str, float] = {
        p: math.log((N + 1) / (f + 1)) + 1.0
        for p, f in kw_doc_freq_map.items()
    }
    kw_idf_max = math.log(N + 1) + 1.0   # IDF maksimum saat doc_freq = 0

    state.df              = df
    state.tfidf_word      = tfidf_word
    state.tfidf_char      = tfidf_char
    state.mat_word        = mat_word
    state.mat_char        = mat_char
    state.bm25            = bm25
    state.kw_freq         = Counter(all_kw)
    state.total_docs      = N
    state.kw_idf          = kw_idf
    state.kw_idf_max      = kw_idf_max
    state.word_in_kw_freq = word_in_kw_freq
    print(f"✅ Model siap — {N} judul, {len(state.kw_freq)} kata kunci unik, "
          f"{len(kw_idf)} frasa IDF, {len(word_in_kw_freq)} kata IDF")

# ══════════════════════════════════════════════════════════════
#  SCORING ENGINE
# ══════════════════════════════════════════════════════════════
def compute_saturation(input_kw_list: list[str]):
    """
    Kejenuhan topik dihitung berdasarkan exact phrase match kata kunci.
    'prestasi belajar' hanya dihitung jenuh jika memang banyak skripsi
    yang punya kata kunci 'prestasi belajar' persis — bukan sekedar
    mengandung kata 'belajar'.
    """
    if not input_kw_list:
        return 0.0, {}
    detail = {}
    for kw in input_kw_list:
        kw_norm = re.sub(r'\s+', ' ', kw.strip().lower())
        # Exact match frasa
        exact   = state.kw_freq.get(kw_norm, 0)
        # Partial: kw_norm terkandung dalam frasa lain di database
        partial = sum(
            cnt for db_kw, cnt in state.kw_freq.items()
            if kw_norm in db_kw or db_kw in kw_norm
        )
        detail[kw] = max(exact, partial) / state.total_docs
    return float(np.mean(list(detail.values()))), detail

def run_search(judul_input: str, kata_kunci_input: str, top_n: int = 15) -> dict:
    input_kw_list  = parse_keywords(kata_kunci_input)
    weighted_query = build_weighted_text(judul_input, kata_kunci_input)
    query_tokens   = weighted_query.split()

    sim_word = cosine_similarity(
        state.tfidf_word.transform([weighted_query]), state.mat_word
    ).flatten()
    sim_char = cosine_similarity(
        state.tfidf_char.transform([weighted_query]), state.mat_char
    ).flatten()
    raw_bm25 = state.bm25.get_scores(query_tokens)
    bm25_max = raw_bm25.max() if raw_bm25.max() > 0 else 1
    sim_bm25 = raw_bm25 / bm25_max
    # ── Hitung Jaccard + match_scale per dokumen ──────────────
    jac_scores = np.zeros(len(state.df))
    jac_scales = np.zeros(len(state.df))
    for i in range(len(state.df)):
        score, scale  = jaccard_kw(input_kw_list, state.df.iloc[i]['KW_List'])
        jac_scores[i] = score
        jac_scales[i] = scale

    # ── Ensemble dengan dynamic keyword weight balancing ───────
    # Bobot keyword efektif per dokumen = W_JACCARD_KW × match_scale
    # Sisa bobot didistribusikan kembali ke metode berbasis judul
    # Total bobot selalu tetap 1.0 per dokumen.
    TITLE_TOTAL = W_TFIDF_WORD + W_TFIDF_CHAR + W_BM25      # 0.30

    effective_kw     = jac_scores * jac_scales * W_JACCARD_KW  # array
    unused_kw_weight = (1.0 - jac_scales) * W_JACCARD_KW       # array

    # Redistribusi ke sinyal judul proporsional dengan bobot asal
    r_word = unused_kw_weight * (W_TFIDF_WORD / TITLE_TOTAL)
    r_char = unused_kw_weight * (W_TFIDF_CHAR / TITLE_TOTAL)
    r_bm25 = unused_kw_weight * (W_BM25       / TITLE_TOTAL)

    ensemble = (
        (W_TFIDF_WORD + r_word) * sim_word +
        (W_TFIDF_CHAR + r_char) * sim_char +
        (W_BM25       + r_bm25) * sim_bm25 +
        effective_kw
    )

    results = []
    for i in np.argsort(ensemble)[::-1]:
        sc = float(ensemble[i])
        if sc < THR_TAMPIL:
            break
        results.append({
            "judul"      : state.df.iloc[i]['Judul'],
            "kata_kunci" : state.df.iloc[i]['Kata_Kunci'],
            "tempat"     : state.df.iloc[i]['Tempat'],
            "skor"       : round(sc * 100, 1),
            "tfidf"      : round(float(sim_word[i]) * 100, 1),
            "bm25"       : round(float(sim_bm25[i]) * 100, 1),
            "jaccard_kw" : round(float(jac_scores[i]) * 100, 1),
            "kw_scale"   : round(float(jac_scales[i]) * 100, 1),
        })
        if len(results) >= top_n:
            break

    avg_sat, detail_sat = compute_saturation(input_kw_list)
    n_sangat_mirip      = sum(1 for r in results if r['skor'] >= THR_SANGAT_MIRIP * 100)
    n_mirip             = len(results)
    topik_jenuh         = (avg_sat >= THR_TOPIK_JENUH_KW) or (n_mirip >= THR_TOPIK_JENUH_N)

    if n_sangat_mirip >= 1:
        status = "TERLALU_MIRIP"
    elif topik_jenuh:
        status = "TOPIK_UMUM"
    else:
        status = "AMAN"

    # ── Spesifisitas tiap keyword input (untuk transparansi ke frontend) ──
    kw_specificity_detail = {
        kw: round(
            phrase_specificity(re.sub(r'\s+', ' ', kw.strip().lower())) * 100, 1
        )
        for kw in input_kw_list
    }

    # Simpan ke log
    log_pencarian(judul_input, kata_kunci_input, status, n_mirip, round(avg_sat * 100, 1))

    return {
        "status"              : status,
        "n_sangat_mirip"      : n_sangat_mirip,
        "n_mirip"             : n_mirip,
        "avg_saturation"      : round(avg_sat * 100, 1),
        "detail_saturation"   : {k: round(v * 100, 1) for k, v in detail_sat.items()},
        "kw_specificity"      : kw_specificity_detail,   # NEW: spesifisitas per keyword
        "total_db"            : state.total_docs,
        "results"             : results,
    }

# ══════════════════════════════════════════════════════════════
#  AUTH HELPER (Admin)
# ══════════════════════════════════════════════════════════════
def verify_admin_token(x_admin_token: str = Header(default="")):
    if x_admin_token not in _valid_tokens:
        raise HTTPException(status_code=401, detail="Token tidak valid. Silakan login admin.")

# ══════════════════════════════════════════════════════════════
#  FASTAPI APP
# ══════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    load_model()
    yield
    print("👋 Server berhenti.")

app = FastAPI(
    title="SiMiJu v3",
    description="Sistem Deteksi Kemiripan Judul Skripsi — STAB Negeri Raden Wijaya Wonogiri",
    version="3.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# ── Pydantic schemas ──────────────────────────────────────────
class CekRequest(BaseModel):
    judul:      str
    kata_kunci: str

class LoginRequest(BaseModel):
    password: str

# ══════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════
@app.post("/api/cek")
async def cek(body: CekRequest):
    judul  = body.judul.strip()
    kw_raw = body.kata_kunci.strip()
    if not judul:
        raise HTTPException(status_code=400, detail="Judul tidak boleh kosong")
    if not kw_raw:
        raise HTTPException(status_code=400, detail="Kata kunci tidak boleh kosong")
    return run_search(judul, kw_raw, top_n=15)

@app.get("/api/stats")
async def stats():
    top_kw = [
        {"kw": kw, "freq": freq, "pct": round(freq / state.total_docs * 100, 1)}
        for kw, freq in state.kw_freq.most_common(20)
    ]
    return {"total": state.total_docs, "top_keywords": top_kw}

@app.get("/api/health")
async def health():
    return {"status": "ok", "total_docs": state.total_docs}

def _normalize_untuk_cari(text: str) -> str:
    """
    Normalisasi string untuk fuzzy matching lokasi:
    - Lowercase
    - Ganti angka-huruf mirip: o→0, l→1 (dan sebaliknya)
    - Hapus karakter non-alfanumerik
    """
    text = text.lower()
    # Normalisasi huruf yang sering salah ketik sebagai angka
    text = re.sub(r'\bo\b', '0', text)       # 'o' standalone → '0'
    text = re.sub(r'(?<=\d)o(?=\d)', '0', text)  # angka-o-angka → angka-0-angka
    text = re.sub(r'(?<=\s)o(?=\d)', '0', text)  # spasi-o-angka → spasi-0-angka
    text = re.sub(r'(?<=\d)o(?=\s)', '0', text)  # angka-o-spasi → angka-0-spasi
    return re.sub(r'[^a-z0-9\s]', ' ', text)

def _cari_cocok_lokasi(q_norm: str, judul: str, tempat: str) -> bool:
    """
    Cek apakah query lokasi cocok dengan tempat atau judul skripsi.
    Menggunakan fuzzy: toleran terhadap typo 0↔o, normalisasi spasi, partial match.
    """
    # Cek di kolom Tempat dulu (sudah diekstrak)
    if tempat and q_norm in _normalize_untuk_cari(tempat):
        return True
    # Fallback: cari langsung di judul (untuk judul yang gagal diekstrak tempat-nya)
    judul_norm = _normalize_untuk_cari(judul)
    if q_norm in judul_norm:
        return True
    # Fuzzy: tiap kata query harus ada di judul (mendukung urutan berbeda)
    q_words = q_norm.split()
    if len(q_words) >= 2 and all(w in judul_norm for w in q_words):
        return True
    return False

@app.get("/api/search-lokasi")
async def search_lokasi(q: str = ""):
    """
    Cari semua skripsi yang berlokasi di tempat yang diberikan.

    Pencocokan toleran (fuzzy):
    - Case-insensitive
    - Toleran typo angka: o1 = 01 = 1
    - Partial match: query cukup mengandung sebagian nama tempat
    - Multi-kata: semua kata query harus ada di kolom Tempat

    Hanya mencari di kolom Tempat (yang sudah diisi dari CSV).
    Judul tanpa data lokasi tidak akan muncul.
    """
    q_clean = q.strip()
    if not q_clean:
        return {"query": "", "total": 0, "results": []}

    q_norm = _normalize_untuk_cari(q_clean)
    q_words = q_norm.split()

    results = []
    for _, row in state.df.iterrows():
        tempat = str(row.get("Tempat", "") or "").strip()
        if not tempat:
            continue
        t_norm = _normalize_untuk_cari(tempat)
        # Cocok jika: partial match ATAU semua kata query ada di tempat
        if q_norm in t_norm or (len(q_words) >= 2 and all(w in t_norm for w in q_words)):
            results.append({
                "judul"      : row["Judul"],
                "kata_kunci" : row["Kata_Kunci"],
                "tempat"     : tempat,
            })

    return {
        "query"  : q_clean,
        "total"  : len(results),
        "results": results,
    }

@app.get("/api/daftar-lokasi")
async def daftar_lokasi(q: str = ""):
    """
    Kembalikan semua lokasi unik yang ada di dataset (kolom Tempat).
    Jika q diberikan, filter hanya yang mengandung string tersebut.
    Digunakan untuk combobox autocomplete di frontend.
    """
    q_norm = _normalize_untuk_cari(q.strip()) if q.strip() else ""

    lokasi_set = set()
    for tempat in state.df["Tempat"]:
        t = str(tempat or "").strip()
        if not t:
            continue
        if not q_norm or q_norm in _normalize_untuk_cari(t):
            lokasi_set.add(t)

    lokasi_sorted = sorted(lokasi_set, key=lambda x: x.lower())
    return {"total": len(lokasi_sorted), "lokasi": lokasi_sorted}

# ══════════════════════════════════════════════════════════════
#  ADMIN API — Prioritas #4
# ══════════════════════════════════════════════════════════════
@app.post("/api/admin/login")
async def admin_login(body: LoginRequest):
    if body.password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Password salah")
    token = secrets.token_hex(32)
    _valid_tokens.add(token)
    return {"token": token}

@app.post("/api/admin/upload-dataset")
async def upload_dataset(
    file: UploadFile = File(...),
    mode: str = "append",          # "append" | "replace"
    _: None = Depends(verify_admin_token)
):
    """
    Upload CSV dataset skripsi.
    mode=append  → data baru ditambahkan ke dataset yang sudah ada (default)
    mode=replace → dataset lama diganti sepenuhnya dengan data baru
    Format CSV: kolom Judul dan Kata_Kunci
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Hanya file CSV yang diterima")

    # ── Baca & validasi file baru ──────────────────────
    content = await file.read()

    def baca_csv(data: bytes) -> pd.DataFrame:
        """
        Baca CSV dengan auto-detect encoding, separator, dan quoting.
        Menangani kasus:
        - Separator koma, titik koma, atau tab
        - Encoding UTF-8, UTF-8 BOM, Latin-1
        - Tanda kutip ganda di dalam nilai (standar CSV Excel)
        - Kata kunci yang diapit tanda kutip
        """
        for enc in ['utf-8-sig', 'utf-8', 'latin-1']:
            for sep in [',', ';', '\t']:
                try:
                    df = pd.read_csv(
                        io.BytesIO(data),
                        encoding=enc,
                        sep=sep,
                        quotechar='"',
                        doublequote=True,    # "" di dalam nilai → " tunggal
                        skipinitialspace=True,
                        on_bad_lines='skip', # baris rusak dilewati
                    )
                    if len(df.columns) >= 2 and len(df) > 0:
                        return df
                except Exception:
                    continue
        raise ValueError("Tidak dapat membaca file CSV. Pastikan format sesuai panduan.")

    def bersihkan_kata_kunci(kw: str) -> str:
        """
        Normalisasi kolom kata kunci:
        - Hapus tanda kutip di awal/akhir
        - Ganti titik koma menjadi koma (sering terjadi dari Excel)
        - Rapikan spasi
        """
        kw = kw.strip().strip('"').strip("'")
        kw = re.sub(r'\s*;\s*', ', ', kw)   # titik koma → koma
        kw = re.sub(r'\s*\|\s*', ', ', kw)  # pipe → koma
        kw = re.sub(r',\s*', ', ', kw)      # rapikan spasi setelah koma
        kw = kw.strip(', ')
        return kw

    try:
        df_new = baca_csv(content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if len(df_new.columns) < 2:
        raise HTTPException(status_code=400, detail="CSV minimal harus punya 2 kolom: Judul dan Kata_Kunci")

    df_new.columns = ['Judul', 'Kata_Kunci'] + list(df_new.columns[2:])
    df_new = df_new[['Judul', 'Kata_Kunci']].copy()
    df_new = df_new.dropna(subset=['Judul'])
    df_new['Judul']      = df_new['Judul'].astype(str).str.strip().str.strip('"')
    df_new['Kata_Kunci'] = df_new['Kata_Kunci'].fillna('').astype(str).apply(bersihkan_kata_kunci)
    df_new = df_new[df_new['Judul'] != '']

    if len(df_new) == 0:
        raise HTTPException(status_code=400, detail="Tidak ada data valid setelah dibersihkan. Pastikan kolom Judul tidak kosong.")

    jumlah_masuk = len(df_new)

    if mode == "append":
        # ── MODE TAMBAH: gabungkan dengan dataset lama ──
        if DATASET_PATH.exists():
            df_lama = pd.read_csv(DATASET_PATH, encoding='utf-8-sig')
            cols_lama = list(df_lama.columns)
            if len(cols_lama) >= 3:
                df_lama.columns = ['Judul', 'Kata_Kunci', 'Tempat'] + cols_lama[3:]
                df_lama = df_lama[['Judul', 'Kata_Kunci', 'Tempat']]
            else:
                df_lama.columns = ['Judul', 'Kata_Kunci'] + cols_lama[2:]
                df_lama = df_lama[['Judul', 'Kata_Kunci']]
                df_lama['Tempat'] = ''
            jumlah_lama = len(df_lama)

            # Gabungkan dan hapus duplikat berdasarkan Judul (case-insensitive)
            df_gabung = pd.concat([df_lama, df_new], ignore_index=True)
            df_gabung['_judul_lower'] = df_gabung['Judul'].str.lower().str.strip()
            df_gabung = df_gabung.drop_duplicates(subset='_judul_lower', keep='first')
            df_gabung = df_gabung.drop(columns=['_judul_lower'])

            jumlah_duplikat = (jumlah_lama + jumlah_masuk) - len(df_gabung)
            df_final = df_gabung
            pesan = (
                f"Berhasil menambahkan data. "
                f"{jumlah_masuk} judul masuk, "
                f"{jumlah_duplikat} duplikat diabaikan, "
                f"total sekarang {len(df_final)} judul."
            )
        else:
            # Belum ada dataset sama sekali, simpan langsung
            df_final = df_new
            jumlah_duplikat = 0
            pesan = f"Dataset baru dibuat dengan {len(df_final)} judul."

    else:
        # ── MODE GANTI: backup lama, simpan yang baru ──
        if DATASET_PATH.exists():
            backup_path = DATASET_PATH.with_suffix(
                f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            )
            import shutil
            shutil.copy2(DATASET_PATH, backup_path)

        df_final = df_new
        jumlah_duplikat = 0
        pesan = f"Dataset berhasil diganti. {len(df_final)} judul dimuat."

    # ── Simpan dataset final (dengan kolom Tempat jika ada) ──
    save_cols = ['Judul', 'Kata_Kunci']
    if 'Tempat' in df_final.columns:
        save_cols.append('Tempat')
    df_final[save_cols].to_csv(DATASET_PATH, index=False, encoding='utf-8-sig')

    # ── Reload model ───────────────────────────────────
    load_model()

    return {
        "message"         : pesan,
        "mode"            : mode,
        "jumlah_masuk"    : jumlah_masuk,
        "jumlah_duplikat" : jumlah_duplikat,
        "total_baru"      : state.total_docs,
    }

@app.get("/api/admin/log")
async def get_log_api(limit: int = 100, _: None = Depends(verify_admin_token)):
    """Ambil riwayat pencarian mahasiswa."""
    return {"log": get_log(limit)}

@app.get("/api/admin/log/summary")
async def log_summary(_: None = Depends(verify_admin_token)):
    """Ringkasan statistik penggunaan sistem."""
    con = sqlite3.connect(DB_PATH)
    total       = con.execute("SELECT COUNT(*) FROM log_pencarian").fetchone()[0]
    per_status  = con.execute(
        "SELECT status, COUNT(*) FROM log_pencarian GROUP BY status"
    ).fetchall()
    per_hari    = con.execute(
        "SELECT DATE(waktu) as hari, COUNT(*) FROM log_pencarian GROUP BY hari ORDER BY hari DESC LIMIT 30"
    ).fetchall()
    con.close()
    return {
        "total_pencarian" : total,
        "per_status"      : {r[0]: r[1] for r in per_status},
        "per_hari"        : [{"tanggal": r[0], "jumlah": r[1]} for r in per_hari],
    }


# ══════════════════════════════════════════════════════════════
#  STATIC + SPA FALLBACK
# ══════════════════════════════════════════════════════════════
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

@app.get("/{full_path:path}", include_in_schema=False)
async def spa_fallback(full_path: str):
    return FileResponse(str(BASE_DIR / "static" / "index.html"))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)