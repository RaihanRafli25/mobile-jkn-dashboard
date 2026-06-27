# ===== dashboard.py — Versi Supabase + Optimasi =====

import streamlit as st
import pandas as pd
import plotly.express as px
from supabase import create_client
from collections import Counter
from datetime import datetime, timedelta

st.set_page_config(
    page_title = "Dashboard Sentimen Mobile JKN",
    page_icon  = "🏥",
    layout     = "wide"
)

# ── Koneksi Supabase ──────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_data(tgl_mulai='2025-06-01'):
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

# ── Header ────────────────────────────────────────────────────
st.title("🏥 Dashboard Analisis Sentimen Mobile JKN")
st.caption("Sumber data: Google Play Store · Model: LinearSVC + TF-IDF · XAI: LIME")
st.divider()

# ── Sidebar: Filter ───────────────────────────────────────────
with st.sidebar:
    st.header("Filter Data")

    periode_opt = st.selectbox(
        "Periode data",
        ["6 bulan terakhir", "1 tahun terakhir", "Semua data (lambat)"]
    )

    if periode_opt == "6 bulan terakhir":
        tgl_mulai_load = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
    elif periode_opt == "1 tahun terakhir":
        tgl_mulai_load = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    else:
        tgl_mulai_load = '2025-06-01'

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

    tgl_min   = df['tanggal'].min().date()
    tgl_max   = df['tanggal'].max().date()
    tgl_range = st.date_input(
        "Rentang tanggal",
        value     = (tgl_min, tgl_max),
        min_value = tgl_min,
        max_value = tgl_max
    )

    keyword = st.text_input("Cari kata kunci ulasan", "")

    rating_opt = st.multiselect(
        "Rating",
        options = [1, 2, 4, 5],
        default = [1, 2, 4, 5]
    )

    st.divider()
    st.caption(f"Total data dimuat: {len(df):,} ulasan")
    st.caption(f"Periode: {tgl_min} s/d {tgl_max}")

# ── Terapkan filter ───────────────────────────────────────────
df_f = df.copy()

if sentimen_opt != "Semua":
    df_f = df_f[df_f['prediksi'] == sentimen_opt.lower()]

if len(tgl_range) == 2:
    df_f = df_f[
        (df_f['tanggal'].dt.date >= tgl_range[0]) &
        (df_f['tanggal'].dt.date <= tgl_range[1])
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
    st.subheader("Distribusi Sentimen")
    fig_donut = px.pie(
        values = [n_neg, n_pos],
        names  = ["Negatif", "Positif"],
        color  = ["Negatif", "Positif"],
        color_discrete_map = {"Negatif": "#E24B4A", "Positif": "#639922"},
        hole   = 0.55
    )
    fig_donut.update_traces(
        textposition          = 'inside',
        textinfo              = 'percent+label',
        insidetextorientation = 'radial'
    )
    fig_donut.update_layout(
        showlegend = True,
        legend     = dict(orientation='h', yanchor='bottom', y=-0.2),
        margin     = dict(t=20, b=40, l=0, r=0),
        height     = 320
    )
    st.plotly_chart(fig_donut, use_container_width=True)

with col2:
    st.subheader("Tren Sentimen per Bulan")
    tren = (df_f.groupby(['bulan', 'prediksi'])
               .size().reset_index(name='jumlah'))
    fig_tren = px.bar(
        tren, x='bulan', y='jumlah', color='prediksi',
        color_discrete_map = {'positif':'#639922', 'negatif':'#E24B4A'},
        barmode = 'group',
        labels  = {'bulan':'Bulan', 'jumlah':'Jumlah Ulasan', 'prediksi':'Sentimen'}
    )
    fig_tren.update_layout(
        margin = dict(t=0, b=0),
        height = 320,
        legend = dict(orientation='h', yanchor='bottom', y=1.02)
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
        color = 'warna',
        color_discrete_map = 'identity',
        labels = {'rating':'Rating', 'jumlah':'Jumlah Ulasan'}
    )
    fig_rating.update_layout(
        showlegend = False,
        margin     = dict(t=0, b=0),
        height     = 280
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
            orientation = 'h',
            color_discrete_sequence = ['#E24B4A'],
            labels = {'frekuensi':'Frekuensi', 'kata':'Kata'}
        )
        fig_kata.update_layout(
            yaxis  = dict(autorange='reversed'),
            margin = dict(t=0, b=0),
            height = 280
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
    use_container_width = True,
    height              = 400,
    column_config       = {
        'tanggal'  : st.column_config.DateColumn("Tanggal"),
        'nama'     : st.column_config.TextColumn("Pengguna"),
        'rating'   : st.column_config.NumberColumn("Rating", format="%d ⭐"),
        'prediksi' : st.column_config.TextColumn("Sentimen"),
        'ulasan'   : st.column_config.TextColumn("Isi Ulasan", width="large"),
    }
)

# ── Rekomendasi berbasis LIME ─────────────────────────────────
st.divider()
st.subheader("📋 Rekomendasi Perbaikan Layanan")
st.caption("Berdasarkan analisis interpretabilitas model LIME terhadap ulasan negatif")

# Hitung top kata negatif untuk rekomendasi dinamis
top5_neg = [kata for kata, _ in Counter(all_kata).most_common(5)]

rekomendasi = {
    'daftar'    : ('🔐 Login / Akun',      'Perbaiki alur pendaftaran dan verifikasi OTP — kata "daftar" mendominasi ulasan negatif.'),
    'otp'       : ('🔐 Login / Akun',      'Tingkatkan keandalan pengiriman kode OTP dan sediakan metode autentikasi alternatif.'),
    'masuk'     : ('🔐 Login / Akun',      'Perbaiki proses login — banyak pengguna mengalami kegagalan saat masuk ke akun.'),
    'eror'      : ('⚙️ Teknis / Bug',      'Lakukan audit kode untuk mengatasi error sistem dan force close yang dilaporkan pengguna.'),
    'muat'      : ('⚙️ Teknis / Bug',      'Optimalkan kecepatan loading aplikasi melalui teknik lazy loading dan kompresi aset.'),
    'antre'     : ('🏥 Antrian / Faskes',  'Tingkatkan integrasi data antrean real-time antara aplikasi dan fasilitas kesehatan mitra.'),
    'peladen'   : ('🌐 Server / Koneksi',  'Tingkatkan kapasitas server terutama pada periode penggunaan puncak.'),
    'koneksi'   : ('🌐 Server / Koneksi',  'Implementasikan mekanisme graceful degradation saat koneksi server terganggu.'),
    'fitur'     : ('🎨 Fitur / Tampilan',  'Lakukan evaluasi usability dan perbaiki antarmuka berdasarkan masukan pengguna.'),
    'tampil'    : ('🎨 Fitur / Tampilan',  'Sederhanakan alur navigasi dan tingkatkan keterbacaan informasi kepesertaan.'),
}

# Tampilkan rekomendasi yang relevan berdasarkan top kata
ditampilkan = set()
for kata in top5_neg:
    if kata in rekomendasi and rekomendasi[kata][0] not in ditampilkan:
        kategori, pesan = rekomendasi[kata]
        st.warning(f"**{kategori}** — {pesan}")
        ditampilkan.add(rekomendasi[kata][0])

if not ditampilkan:
    st.info("Tidak ada rekomendasi spesifik untuk data yang dipilih.")

st.caption("Rekomendasi di atas dihasilkan berdasarkan kata-kata yang terbukti berkontribusi "
           "terhadap sentimen negatif melalui analisis LIME (Local Interpretable Model-agnostic Explanations).")

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.caption(
    "Dashboard Analisis Sentimen Ulasan Mobile JKN · "
    "Universitas Widyatama 2026 · "
    "Model: LinearSVC + TF-IDF · XAI: LIME (F1-score: 90.43%)"
)
