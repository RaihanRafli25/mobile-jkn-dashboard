# ===== dashboard.py — Versi Final dengan Rekomendasi Dinamis LIME =====

import streamlit as st
import pandas as pd
import plotly.express as px
from supabase import create_client
from collections import Counter
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Dashboard Sentimen Mobile JKN",
    page_icon="🏥",
    layout="wide"
)

# ── Load koefisien SVM ────────────────────────────────────────
@st.cache_data
def load_coef():
    df = pd.read_csv('svm_coefficients.csv')
    return dict(zip(df['kata'], df['bobot']))

# ── Koneksi Supabase ──────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_data(tgl_mulai='2024-01-01'):
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    sb  = create_client(url, key)

    all_data = []
    page     = 0
    while True:
        res = sb.table('ulasan').select('*').gte(
            'tanggal', tgl_mulai
        ).range(
            page * 1000, (page + 1) * 1000 - 1
        ).execute()
        if not res.data:
            break
        all_data.extend(res.data)
        if len(res.data) < 1000:
            break
        page += 1

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df['tanggal'] = pd.to_datetime(df['tanggal'])
    df['bulan']   = df['tanggal'].dt.to_period('M').astype(str)
    return df

# ── Fungsi generate teks dinamis ─────────────────────────────
def generate_teks_rekomendasi(kata, frekuensi, persen, total_neg, periode):
    template = {
'nomor': (
            '🔐 Login / Akun',
            f'Pada periode {periode}, kata **"nomor"** muncul **{frekuensi:,} kali '
            f'({persen:.1f}% ulasan negatif)**, menunjukkan permasalahan terkait '
            f'verifikasi nomor telepon yang digunakan untuk autentikasi akun. '
            f'**Rekomendasi:** Perbaiki sistem validasi nomor telepon dan sediakan '
            f'opsi penggantian nomor yang terdaftar melalui verifikasi data kepesertaan BPJS.'
        ),
        'susah': (
            '⚙️ Teknis / Bug',
            f'Pada periode {periode}, kata **"susah"** terdeteksi **{frekuensi:,} kali '
            f'({persen:.1f}% ulasan negatif)**, mengindikasikan pengalaman pengguna '
            f'yang menyulitkan secara umum pada berbagai fitur aplikasi. '
            f'**Rekomendasi:** Lakukan user testing menyeluruh untuk mengidentifikasi '
            f'titik-titik kesulitan utama dan sederhanakan alur penggunaan fitur kritis.'
        ),
        'ribet': (
            '🎨 Fitur / Tampilan',
            f'Pada periode {periode}, kata **"ribet"** muncul **{frekuensi:,} kali '
            f'({persen:.1f}% ulasan negatif)**, menunjukkan bahwa pengguna merasa '
            f'alur penggunaan aplikasi terlalu rumit. '
            f'**Rekomendasi:** Sederhanakan alur navigasi, kurangi jumlah langkah '
            f'untuk menyelesaikan tugas utama, dan redesign UI mengikuti prinsip UX modern.'
        ),
        'rujuk': (
            '🏥 Antrian / Faskes',
            f'Pada periode {periode}, kata **"rujuk"** terdeteksi **{frekuensi:,} kali '
            f'({persen:.1f}% ulasan negatif)**. Proses rujukan online dinilai '
            f'tidak berjalan sesuai harapan pengguna. '
            f'**Rekomendasi:** Sederhanakan proses pengajuan rujukan online dan pastikan '
            f'data faskes penerima rujukan selalu diperbarui secara berkala.'
        ),
        'dokter': (
            '🏥 Antrian / Faskes',
            f'Pada periode {periode}, kata **"dokter"** muncul **{frekuensi:,} kali '
            f'({persen:.1f}% ulasan negatif)**, mengindikasikan keluhan terkait '
            f'informasi atau ketersediaan dokter di fasilitas kesehatan mitra. '
            f'**Rekomendasi:** Perbarui informasi ketersediaan dokter secara real-time '
            f'dan sediakan fitur pemilihan dokter yang lebih transparan di aplikasi.'
        ),
        'antrean': (
            '🏥 Antrian / Faskes',
            f'Pada periode {periode}, kata **"antrean"** terdeteksi **{frekuensi:,} kali '
            f'({persen:.1f}% ulasan negatif)**, menunjukkan ketidaksesuaian antara '
            f'sistem antrean digital dan pelayanan aktual di fasilitas kesehatan. '
            f'**Rekomendasi:** Tingkatkan integrasi data antrean real-time antara aplikasi '
            f'dan sistem informasi fasilitas kesehatan mitra.'
        ),
        'turun': (
            '🌐 Server / Koneksi',
            f'Pada periode {periode}, kata **"turun"** terdeteksi **{frekuensi:,} kali '
            f'({persen:.1f}% ulasan negatif)**, menunjukkan laporan server down '
            f'yang dirasakan pengguna. '
            f'**Rekomendasi:** Tingkatkan keandalan infrastruktur server, implementasikan '
            f'monitoring uptime otomatis, dan sediakan halaman status layanan yang '
            f'dapat diakses pengguna saat terjadi gangguan.'
        ),
        'ganggu': (
            '🌐 Server / Koneksi',
            f'Pada periode {periode}, kata **"ganggu"** muncul **{frekuensi:,} kali '
            f'({persen:.1f}% ulasan negatif)**, mengindikasikan gangguan layanan '
            f'yang dirasakan pengguna secara berulang. '
            f'**Rekomendasi:** Implementasikan sistem notifikasi gangguan proaktif '
            f'kepada pengguna dan tingkatkan prosedur pemulihan layanan yang lebih cepat.'
        ),
        'notifikasi': (
            '🎨 Fitur / Tampilan',
            f'Pada periode {periode}, kata **"notifikasi"** terdeteksi **{frekuensi:,} kali '
            f'({persen:.1f}% ulasan negatif)**, menunjukkan permasalahan pada '
            f'sistem notifikasi aplikasi yang tidak berfungsi optimal. '
            f'**Rekomendasi:** Perbaiki sistem push notification, berikan opsi '
            f'pengaturan notifikasi yang fleksibel, dan pastikan notifikasi penting '
            f'tersampaikan tepat waktu.'
        ),
        'pembaruan': (
            '🎨 Fitur / Tampilan',
            f'Pada periode {periode}, kata **"pembaruan"** muncul **{frekuensi:,} kali '
            f'({persen:.1f}% ulasan negatif)**, mengindikasikan ketidakpuasan '
            f'pengguna terhadap pembaruan aplikasi yang dirilis. '
            f'**Rekomendasi:** Pastikan setiap pembaruan aplikasi melalui pengujian '
            f'menyeluruh sebelum dirilis dan sediakan catatan pembaruan yang jelas.'
        ),
        'nik': (
            '🔐 Login / Akun',
            f'Pada periode {periode}, kata **"nik"** muncul **{frekuensi:,} kali '
            f'({persen:.1f}% ulasan negatif)**, menunjukkan permasalahan terkait '
            f'verifikasi NIK dalam proses pendaftaran atau pemulihan akun. '
            f'**Rekomendasi:** Perbaiki sistem validasi NIK dan sinkronisasi data '
            f'kepesertaan BPJS agar proses verifikasi identitas berjalan lebih lancar.'
        ),
        'sms': (
            '🔐 Login / Akun',
            f'Pada periode {periode}, kata **"sms"** terdeteksi **{frekuensi:,} kali '
            f'({persen:.1f}% ulasan negatif)**, mengindikasikan kegagalan '
            f'pengiriman SMS OTP yang dikeluhkan banyak pengguna. '
            f'**Rekomendasi:** Evaluasi dan ganti gateway SMS yang tidak andal, '
            f'tambahkan mekanisme retry otomatis, dan sediakan OTP alternatif via email.'
        ),
        'kirim': (
            '🔐 Login / Akun',
            f'Pada periode {periode}, kata **"kirim"** muncul **{frekuensi:,} kali '
            f'({persen:.1f}% ulasan negatif)**, menunjukkan kegagalan pengiriman '
            f'kode verifikasi yang dirasakan pengguna. '
            f'**Rekomendasi:** Tingkatkan keandalan sistem pengiriman kode verifikasi '
            f'dan tambahkan konfirmasi status pengiriman yang transparan kepada pengguna.'
        ),
    }

    return template.get(kata, None)


def generate_rekomendasi(df_negatif, coef_dict, periode):
    all_kata  = ' '.join(df_negatif['teks_bersih'].astype(str)).split()
    freq_dict = Counter(all_kata)

    if not freq_dict:
        return []

    skor = {}
    for kata, freq in freq_dict.items():
        bobot = coef_dict.get(kata, 0)
        if bobot < 0:
            skor[kata] = freq * abs(bobot)

    if not skor:
        return []

    top_kata   = sorted(skor.items(), key=lambda x: x[1], reverse=True)[:15]
    hasil      = []
    kategori_x = set()
    total      = len(df_negatif)

    for kata, skor_val in top_kata:
        freq   = freq_dict.get(kata, 0)
        persen = freq / total * 100 if total > 0 else 0
        result = generate_teks_rekomendasi(kata, freq, persen, total, periode)
        if result:
            kat, teks = result
            if kat not in kategori_x:
                hasil.append({
                    'kategori'  : kat,
                    'kata_kunci': kata,
                    'teks'      : teks,
                    'frekuensi' : freq,
                    'persen'    : persen,
                })
                kategori_x.add(kat)

    return hasil


# ── Header ────────────────────────────────────────────────────
st.title("🏥 Dashboard Analisis Sentimen Mobile JKN")
st.caption("Sumber data: Google Play Store · Model: LinearSVC + TF-IDF · XAI: LIME")
st.divider()

# ── Sidebar: Filter ───────────────────────────────────────────
with st.sidebar:
    st.header("Filter Data")

    periode_opt = st.selectbox(
        "Periode data",
        ["6 bulan terakhir", "1 tahun terakhir", "Semua data"]
    )

    # FIX: tanggal mulai load disesuaikan dari hari ini ke belakang
    today = datetime.now()
    if periode_opt == "6 bulan terakhir":
        tgl_mulai_load = (today - timedelta(days=180)).strftime('%Y-%m-%d')
    elif periode_opt == "1 tahun terakhir":
        tgl_mulai_load = (today - timedelta(days=365)).strftime('%Y-%m-%d')
    else:
        tgl_mulai_load = '2024-01-01'

    with st.spinner("Memuat data..."):
        try:
            df = load_data(tgl_mulai=tgl_mulai_load)
        except Exception as e:
            st.error(f"Gagal memuat data: {e}")
            st.stop()

    if df.empty:
        st.warning("Tidak ada data untuk periode ini.")
        st.stop()

    st.divider()

    sentimen_opt = st.selectbox(
        "Sentimen",
        ["Semua", "Positif", "Negatif"]
    )

    tgl_min = df['tanggal'].min().date()
    tgl_max = df['tanggal'].max().date()

    # FIX: pakai list bukan tuple, dan tambah key biar state stabil
    tgl_range = st.date_input(
        "Rentang tanggal",
        value=[tgl_min, tgl_max],
        min_value=tgl_min,
        max_value=tgl_max,
        key="tgl_range"
    )

    keyword = st.text_input("Cari kata kunci ulasan", "")

    rating_opt = st.multiselect(
        "Rating",
        options=[1, 2, 4, 5],
        default=[1, 2, 4, 5]
    )

    st.divider()
    st.caption(f"Total data dimuat: {len(df):,} ulasan")
    st.caption(f"Periode: {tgl_min} s/d {tgl_max}")


# ── Terapkan filter ───────────────────────────────────────────
df_f = df.copy()

if sentimen_opt != "Semua":
    df_f = df_f[df_f['prediksi'] == sentimen_opt.lower()]

# FIX: handle date_input yang bisa return 1 atau 2 elemen saat user lagi milih
if isinstance(tgl_range, (list, tuple)) and len(tgl_range) == 2:
    tgl_start, tgl_end = tgl_range[0], tgl_range[1]
else:
    tgl_start = tgl_range[0] if isinstance(tgl_range, (list, tuple)) else tgl_range
    tgl_end   = tgl_start

df_f = df_f[
    (df_f['tanggal'].dt.date >= tgl_start) &
    (df_f['tanggal'].dt.date <= tgl_end)
]

if keyword:
    df_f = df_f[df_f['ulasan'].str.contains(keyword, case=False, na=False)]

if rating_opt:
    df_f = df_f[df_f['rating'].isin(rating_opt)]


# ── Metric cards ──────────────────────────────────────────────
total   = len(df_f)
n_pos   = (df_f['prediksi'] == 'positif').sum()
n_neg   = (df_f['prediksi'] == 'negatif').sum()
pct_pos = n_pos / total * 100 if total > 0 else 0
pct_neg = n_neg / total * 100 if total > 0 else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total ulasan",      f"{total:,}")
c2.metric("Sentimen positif",  f"{n_pos:,}", f"{pct_pos:.1f}%")
c3.metric("Sentimen negatif",  f"{n_neg:,}", f"{pct_neg:.1f}%")
c4.metric("Akurasi model SVM", "90,43%",     "F1-score")

st.divider()

# ── Baris 1: Donut + Tren bulanan ─────────────────────────────
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Distribusi Sentimen Donut Chart")
    fig_donut = px.pie(
        values=[n_neg, n_pos],
        names=["Negatif", "Positif"],
        color=["Negatif", "Positif"],
        color_discrete_map={"Negatif": "#E24B4A", "Positif": "#639922"},
        hole=0.55
    )
    fig_donut.update_traces(
        textposition='inside',
        textinfo='percent+label',
        insidetextorientation='radial'
    )
    fig_donut.update_layout(
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=-0.2),
        margin=dict(t=20, b=40, l=0, r=0),
        height=320
    )
    st.plotly_chart(fig_donut, use_container_width=True)

with col2:
    st.subheader("Tren Sentimen per Bulan")
    tren = (df_f.groupby(['bulan', 'prediksi'])
               .size().reset_index(name='jumlah'))
    fig_tren = px.bar(
        tren, x='bulan', y='jumlah', color='prediksi',
        color_discrete_map={'positif': '#639922', 'negatif': '#E24B4A'},
        barmode='group',
        labels={'bulan': 'Bulan', 'jumlah': 'Jumlah Ulasan', 'prediksi': 'Sentimen'}
    )
    fig_tren.update_layout(
        margin=dict(t=0, b=0),
        height=320,
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    st.plotly_chart(fig_tren, use_container_width=True)

# ── Baris 2: Distribusi rating + Top kata ─────────────────────
col3, col4 = st.columns(2)

with col3:
    st.subheader("Distribusi Rating")
    rating_count = df_f['rating'].value_counts().sort_index().reset_index()
    rating_count.columns = ['rating', 'jumlah']
    rating_count['warna'] = rating_count['rating'].apply(
        lambda r: '#E24B4A' if r <= 2 else '#639922'
    )
    fig_rating = px.bar(
        rating_count, x='rating', y='jumlah',
        color='warna',
        color_discrete_map='identity',
        labels={'rating': 'Rating', 'jumlah': 'Jumlah Ulasan'}
    )
    fig_rating.update_layout(
        showlegend=False,
        margin=dict(t=0, b=0),
        height=280
    )
    st.plotly_chart(fig_rating, use_container_width=True)

with col4:
    st.subheader("Top 15 Kata pada Ulasan Negatif")
    df_neg   = df_f[df_f['prediksi'] == 'negatif']
    all_kata = ' '.join(df_neg['teks_bersih'].astype(str)).split()
    top_kata = Counter(all_kata).most_common(15)
    if top_kata:
        kata_df  = pd.DataFrame(top_kata, columns=['kata', 'frekuensi'])
        fig_kata = px.bar(
            kata_df, x='frekuensi', y='kata',
            orientation='h',
            color_discrete_sequence=['#E24B4A'],
            labels={'frekuensi': 'Frekuensi', 'kata': 'Kata'}
        )
        fig_kata.update_layout(
            yaxis=dict(autorange='reversed'),
            margin=dict(t=0, b=0),
            height=280
        )
        st.plotly_chart(fig_kata, use_container_width=True)
    else:
        st.info("Tidak ada data untuk ditampilkan.")

# ── Tabel ulasan ──────────────────────────────────────────────
st.divider()
st.subheader(f"Tabel Ulasan ({total:,} data)")

tampil_cols = ['tanggal', 'nama', 'rating', 'prediksi', 'ulasan']
st.dataframe(
    df_f[tampil_cols].sort_values('tanggal', ascending=False),
    use_container_width=True,
    height=400,
    column_config={
        'tanggal' : st.column_config.DateColumn("Tanggal"),
        'nama'    : st.column_config.TextColumn("Pengguna"),
        'rating'  : st.column_config.NumberColumn("Rating", format="%d ⭐"),
        'prediksi': st.column_config.TextColumn("Sentimen"),
        'ulasan'  : st.column_config.TextColumn("Isi Ulasan", width="large"),
    }
)

# ── Rekomendasi Dinamis berbasis LIME + SVM ───────────────────
st.divider()
st.subheader("📋 Rekomendasi Perbaikan Layanan")

# FIX: pakai tgl_start / tgl_end yang sudah aman
periode_label = f"{tgl_start.strftime('%d %b %Y')} – {tgl_end.strftime('%d %b %Y')}"

st.caption(
    f"Dihasilkan secara dinamis berdasarkan **{n_neg:,} ulasan negatif** "
    f"pada periode **{periode_label}** — "
    f"menggunakan bobot koefisien model SVM (pendekatan proxy LIME)"
)

coef_dict  = load_coef()
df_neg_fil = df_f[df_f['prediksi'] == 'negatif']
rekomen    = generate_rekomendasi(df_neg_fil, coef_dict, periode_label)

if rekomen:
    for i, r in enumerate(rekomen, 1):
        with st.expander(
            f"{r['kategori']} — Kata kunci: **\"{r['kata_kunci']}\"** "
            f"({r['frekuensi']:,} kemunculan · {r['persen']:.1f}% ulasan negatif)",
            expanded=(i == 1)
        ):
            st.markdown(r['teks'])
else:
    st.info("Tidak ada rekomendasi spesifik untuk filter yang dipilih saat ini.")

st.caption(
    "⚠️ Rekomendasi di atas bersifat dinamis dan berubah otomatis "
    "mengikuti filter periode, sentimen, dan kata kunci yang aktif."
)
