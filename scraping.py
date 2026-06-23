# ===== scraping.py — Versi Supabase =====
# Jalankan file ini di lokal setiap kali ingin update data
# Data baru akan DITAMBAHKAN (akumulasi), bukan menghapus data lama

import pandas as pd
import re
import string
import time
import joblib
import os
from datetime import date
from google_play_scraper import reviews, Sort
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from supabase import create_client

# ── KONFIGURASI — sesuaikan bagian ini ────────────────────────
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
MODEL_PATH   = "svm_best_model.pkl"
TFIDF_PATH   = "tfidf_vectorizer.pkl"
APP_ID       = "app.bpjs.mobile"
STOP_DATE    = date(2025, 6, 1)
# ──────────────────────────────────────────────────────────────

# ── Koneksi Supabase ──────────────────────────────────────────
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── Cek tanggal terakhir di database ─────────────────────────
# Agar tidak mengambil ulasan yang sudah ada di database
def get_last_date():
    try:
        res = supabase.table('ulasan').select('tanggal').order(
            'tanggal', desc=True).limit(1).execute()
        if res.data:
            last = res.data[0]['tanggal']
            print(f"📅 Data terakhir di database: {last}")
            return date.fromisoformat(last)
        else:
            print("📅 Database kosong, ambil dari STOP_DATE")
            return STOP_DATE
    except:
        return STOP_DATE

# ── Inisialisasi preprocessing ────────────────────────────────
sw_factory      = StopWordRemoverFactory()
sastrawi_sw     = set(sw_factory.get_stop_words())
custom_sw       = {"aplikasi","mobile","jkn","bpjs","kesehatan","app","apk"}
all_stopwords   = sastrawi_sw.union(custom_sw)
stemmer_factory = StemmerFactory()
stemmer         = stemmer_factory.create_stemmer()

normalization_dict = {
    "gk":"tidak","ga":"tidak","gak":"tidak","nggak":"tidak","tdk":"tidak",
    "yg":"yang","dgn":"dengan","utk":"untuk","krn":"karena","jg":"juga",
    "tp":"tapi","blm":"belum","udh":"sudah","sdh":"sudah","bs":"bisa",
    "sy":"saya","gue":"saya","gw":"saya","bgt":"banget","msh":"masih",
    "app":"aplikasi","apk":"aplikasi","login":"masuk","error":"eror",
    "update":"pembaruan","loading":"memuat","bug":"kutu","otp":"otp",
    "hp":"ponsel","notif":"notifikasi","password":"kata sandi",
}

def preprocess(text):
    text = str(text)
    text = re.sub(r'http\S+|https\S+|www\.\S+', ' ', text)
    text = re.sub(r'@\w+|#\w+', ' ', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip().lower()
    words = text.split()
    words = [normalization_dict.get(w, w) for w in words]
    words = [w for w in words if w not in all_stopwords]
    words = [stemmer.stem(w) for w in words]
    return ' '.join(words)

# ── Step 1: Cek tanggal terakhir di database ──────────────────
last_date = get_last_date()
print(f"🔍 Akan mengambil ulasan setelah: {last_date}\n")

# ── Step 2: Scraping ──────────────────────────────────────────
print("🚀 Mulai scraping...")
all_reviews        = []
continuation_token = None
batch_num          = 0
stop_reached       = False

while not stop_reached:
    batch_num += 1
    try:
        result, continuation_token = reviews(
            APP_ID, lang='id', country='id',
            sort=Sort.NEWEST, count=200,
            continuation_token=continuation_token
        )
        if not result:
            break

        for r in result:
            tgl = r['at'].date() if hasattr(r['at'], 'date') else r['at']
            if tgl <= last_date:   # stop jika sudah sampai data lama
                stop_reached = True
                break
            all_reviews.append(r)

        print(f"  Batch {batch_num} | Total baru: {len(all_reviews):,}")

        if continuation_token is None:
            break
        time.sleep(0.5)

    except Exception as e:
        print(f"  Error: {e} — retry 5 detik...")
        time.sleep(5)
        continue

if len(all_reviews) == 0:
    print("✅ Tidak ada ulasan baru. Database sudah up to date.")
    exit()

print(f"✅ Scraping selesai: {len(all_reviews):,} ulasan baru")

# ── Step 3: Cleaning & labeling ───────────────────────────────
df = pd.DataFrame(all_reviews)[['content','score','at','userName']]
df.rename(columns={'content':'ulasan','score':'rating',
                   'at':'tanggal','userName':'nama'}, inplace=True)
df['tanggal'] = pd.to_datetime(df['tanggal']).dt.date.astype(str)
df.drop_duplicates(subset='ulasan', inplace=True)
df.dropna(subset=['ulasan'], inplace=True)
df = df[df['ulasan'].str.strip() != '']
df = df[df['rating'] != 3].copy()
df['label'] = df['rating'].apply(
    lambda r: 'positif' if r >= 4 else 'negatif'
)

# ── Step 4: Preprocessing ─────────────────────────────────────
print("⏳ Preprocessing...")
df['teks_bersih'] = df['ulasan'].apply(preprocess)
df = df[df['teks_bersih'].str.strip() != ''].copy()
df.reset_index(drop=True, inplace=True)
print(f"✅ Preprocessing selesai: {len(df):,} ulasan")

# ── Step 5: Prediksi SVM ──────────────────────────────────────
print("🤖 Prediksi sentimen...")
model    = joblib.load(MODEL_PATH)
tfidf    = joblib.load(TFIDF_PATH)
X        = tfidf.transform(df['teks_bersih'])
df['prediksi'] = model.predict(X)
df['prediksi']  = df['prediksi'].map({1: 'positif', 0: 'negatif'})

# ── Step 6: Simpan ke Supabase (akumulasi) ────────────────────
print("📤 Menyimpan ke Supabase...")
cols = ['tanggal','ulasan','teks_bersih','rating','label','prediksi','nama']
data = df[cols].to_dict(orient='records')

# Upload per batch 500 baris
for i in range(0, len(data), 500):
    batch = data[i:i+500]
    supabase.from_('ulasan').insert(batch).execute()
    print(f"  Upload: {min(i+500, len(data)):,} / {len(data):,}")
    time.sleep(0.5)

print(f"\n✅ Selesai! {len(df):,} ulasan baru berhasil disimpan ke Supabase.")
print(f"   Data lama tetap aman — tidak ada yang dihapus.")
