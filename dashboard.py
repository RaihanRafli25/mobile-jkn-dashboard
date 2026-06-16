# ===== dashboard.py =====
# Deploy file ini ke Streamlit Cloud

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from collections import Counter
import json

st.set_page_config(
    page_title = "Dashboard Sentimen Mobile JKN",
    page_icon  = "🏥",
    layout     = "wide"
)

# ── Load data dari Google Sheets ──────────────────────────────
# Di Streamlit Cloud, credentials disimpan di st.secrets
@st.cache_data(ttl=3600)   # cache 1 jam agar tidak reload terus
def load_data():
    scope  = ["https://spreadsheets.google.com/feeds",
              "https://www.googleapis.com/auth/drive"]
    creds_dict = json.loads(st.secrets["GOOGLE_CREDENTIALS"])
    creds  = ServiceAccountCredentials.from_json_keyfile_dict(
                 creds_dict, scope)
    client = gspread.authorize(creds)
    sheet  = client.open("database_mobile_jkn").sheet1
    data   = sheet.get_all_records()
    df     = pd.DataFrame(data)
    df['tanggal'] = pd.to_datetime(df['tanggal'])
    df['bulan']   = df['tanggal'].dt.to_period('M').astype(str)
    return df

# ── Header ────────────────────────────────────────────────────
st.title("🏥 Dashboard Analisis Sentimen Mobile JKN")
st.caption("Sumber data: Google Play Store · Model: SVM + TF-IDF")
st.divider()

# ── Load data ─────────────────────────────────────────────────
with st.spinner("Memuat data dari database..."):
    try:
        df = load_data()
    except Exception as e:
        st.error(f"Gagal memuat data: {e}")
        st.stop()

# ── Sidebar: Filter ───────────────────────────────────────────
with st.sidebar:
    st.header("Filter Data")

    sentimen_opt = st.selectbox(
        "Sentimen",
        ["Semua", "Positif", "Negatif"]
    )

    tgl_min = df['tanggal'].min().date()
    tgl_max = df['tanggal'].max().date()
    tgl_range = st.date_input(
        "Rentang tanggal",
        value=(tgl_min, tgl_max),
        min_value=tgl_min,
        max_value=tgl_max
    )

    keyword = st.text_input("Cari kata kunci ulasan", "")

    rating_opt = st.multiselect(
        "Rating",
        options=[1, 2, 4, 5],
        default=[1, 2, 4, 5]
    )

    st.divider()
    st.caption(f"Total data: {len(df):,} ulasan")
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
c4.metric("Akurasi model SVM", "90,43%", "F1-score")

st.divider()

# ── Baris 1: Donut + Tren bulanan ─────────────────────────────
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Distribusi sentimen")
    fig_donut = px.pie(
        values=[n_pos, n_neg],
        names=["Positif", "Negatif"],
        color_discrete_sequence=["#639922", "#E24B4A"],
        hole=0.55
    )
    fig_donut.update_traces(textposition='outside', textinfo='percent+label')
    fig_donut.update_layout(
        showlegend=False, margin=dict(t=0, b=0, l=0, r=0), height=280
    )
    st.plotly_chart(fig_donut, use_container_width=True)

with col2:
    st.subheader("Tren sentimen per bulan")
    tren = (df_f.groupby(['bulan', 'prediksi'])
               .size().reset_index(name='jumlah'))
    fig_tren = px.bar(
        tren, x='bulan', y='jumlah', color='prediksi',
        color_discrete_map={'positif':'#639922','negatif':'#E24B4A'},
        barmode='group',
        labels={'bulan':'Bulan','jumlah':'Jumlah ulasan','prediksi':'Sentimen'}
    )
    fig_tren.update_layout(
        margin=dict(t=0, b=0), height=280,
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    st.plotly_chart(fig_tren, use_container_width=True)

# ── Baris 2: Distribusi rating + Kategori keluhan ─────────────
col3, col4 = st.columns(2)

with col3:
    st.subheader("Distribusi rating")
    rating_count = df_f['rating'].value_counts().sort_index().reset_index()
    rating_count.columns = ['rating', 'jumlah']
    rating_count['warna'] = rating_count['rating'].apply(
        lambda r: '#E24B4A' if r <= 2 else '#639922'
    )
    fig_rating = px.bar(
        rating_count, x='rating', y='jumlah',
        color='warna', color_discrete_map='identity',
        labels={'rating':'Rating','jumlah':'Jumlah ulasan'}
    )
    fig_rating.update_layout(
        showlegend=False, margin=dict(t=0, b=0), height=260
    )
    st.plotly_chart(fig_rating, use_container_width=True)

with col4:
    st.subheader("Top 15 kata pada ulasan negatif")
    df_neg   = df_f[df_f['prediksi'] == 'negatif']
    all_kata = ' '.join(df_neg['teks_bersih'].astype(str)).split()
    top_kata = Counter(all_kata).most_common(15)
    if top_kata:
        kata_df  = pd.DataFrame(top_kata, columns=['kata','frekuensi'])
        fig_kata = px.bar(
            kata_df, x='frekuensi', y='kata',
            orientation='h',
            color_discrete_sequence=['#E24B4A'],
            labels={'frekuensi':'Frekuensi','kata':'Kata'}
        )
        fig_kata.update_layout(
            yaxis=dict(autorange='reversed'),
            margin=dict(t=0, b=0), height=260
        )
        st.plotly_chart(fig_kata, use_container_width=True)
    else:
        st.info("Tidak ada data untuk ditampilkan.")

# ── Tabel ulasan ──────────────────────────────────────────────
st.divider()
st.subheader(f"Tabel ulasan ({total:,} data)")

tampil_cols = ['tanggal','nama','rating','prediksi','ulasan']
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

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.caption(
    "Dashboard Analisis Sentimen Ulasan Mobile JKN · "
    "Universitas Widyatama 2026 · "
    "Model: LinearSVC + TF-IDF (F1-score: 90.43%)"
)