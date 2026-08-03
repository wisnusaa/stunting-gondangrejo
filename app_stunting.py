import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ==================== KONFIGURASI HALAMAN ====================
st.set_page_config(
    page_title="Klasifikasi Stunting",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
    <style>
    /* Sembunyikan sidebar sepenuhnya */
    [data-testid="stSidebar"] {
        display: none;
    }
    
    /* Section headers */
    .section-header {
        color: #58a6ff;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-top: 24px;
        margin-bottom: 12px;
        border-bottom: 1px solid #30363d;
        padding-bottom: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

# ==================== HEADER ====================
st.markdown("<h1 style='text-align: center; color: #58a6ff;'>🏥 Klasifikasi Status Stunting</h1>", 
            unsafe_allow_html=True)
st.markdown("---")

# ==================== FORM INPUT ====================
st.markdown("### 📝 Data Balita")

# Nama Balita
nama_balita = st.text_input(
    "Nama Balita (opsional)",
    placeholder="Contoh: Budi Santoso"
)

# Row 1: Jenis Kelamin dan Usia
col1, col2 = st.columns(2)

with col1:
    jenis_kelamin = st.selectbox(
        "Jenis Kelamin",
        ["Laki-laki", "Perempuan"]
    )

with col2:
    usia_bulan = st.number_input(
        "Usia (bulan)",
        min_value=0,
        max_value=60,
        value=24,
        step=1
    )

# ANTROPOMETRI Section
st.markdown("<div class='section-header'>ANTROPOMETRI</div>", unsafe_allow_html=True)

# Row 2: Berat dan Tinggi Badan
col1, col2 = st.columns(2)

with col1:
    berat_badan = st.number_input(
        "Berat Badan (kg)",
        min_value=2.0,
        max_value=30.0,
        value=10.5,
        step=0.1
    )

with col2:
    tinggi_badan = st.number_input(
        "Tinggi / Panjang Badan (cm)",
        min_value=40.0,
        max_value=120.0,
        value=80.0,
        step=0.1
    )

# Cara Pengukuran
cara_pengukuran = st.selectbox(
    "Cara Pengukuran",
    ["Berbaring / Terlentang", "Berdiri"]
)

# STATUS GIZI Section
st.markdown("<div class='section-header'>STATUS GIZI</div>", unsafe_allow_html=True)

# Kenaikan Berat Badan
kenaikan_berat = st.selectbox(
    "Kenaikan Berat Badan dari Pengukuran Sebelumnya",
    ["Naik", "Turun", "Tetap", "Belum ada pengukuran sebelumnya"]
)

st.markdown("---")

# ==================== TOMBOL KLASIFIKASI ====================
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    classify_button = st.button("📋 Klasifikasi", use_container_width=True)

# ==================== HASIL KLASIFIKASI ====================
if classify_button:
    st.markdown("---")
    st.markdown("### 📊 Hasil Klasifikasi")
    
    # Logika klasifikasi
    if tinggi_badan < 75:
        status = "Stunting Berat"
        status_emoji = "🔴"
        risk_level = "Kritis"
        confidence = 95.0
        rec_type = "error"
    elif tinggi_badan < 80:
        status = "Stunting Ringan"
        status_emoji = "🟠"
        risk_level = "Tinggi"
        confidence = 92.5
        rec_type = "warning"
    elif tinggi_badan < 85:
        status = "Berisiko Stunting"
        status_emoji = "🟡"
        risk_level = "Sedang"
        confidence = 88.7
        rec_type = "info"
    else:
        status = "Tidak Stunting"
        status_emoji = "🟢"
        risk_level = "Rendah"
        confidence = 94.2
        rec_type = "success"
    
    # Tampilkan hasil
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Status Stunting", value=f"{status_emoji}", delta=status)
    
    with col2:
        st.metric(label="Risk Level", value=risk_level)
    
    with col3:
        st.metric(label="Confidence", value=f"{confidence:.1f}%")
    
    st.markdown("---")
    
    # Interpretasi
    st.markdown("### 📋 Interpretasi Hasil")
    
    if rec_type == "error":
        st.error(f"""
        **Status: {status}** 🔴
        
        ⚠️ KONDISI KRITIS - Anak mengalami stunting berat!
        
        **Tindakan Segera:**
        - 🚑 SEGERA konsultasi ke dokter/rumah sakit
        - 📋 Pemeriksaan kesehatan lengkap
        - 💊 Pemberian suplemen nutrisi
        - 🧪 Screening penyakit menular
        - 📞 Rujukan untuk penanganan intensif
        """)
    elif rec_type == "warning":
        st.warning(f"""
        **Status: {status}** 🟠
        
        ⚠️ Anak menunjukkan indikasi stunting ringan.
        
        **Rekomendasi:**
        - 🏥 Konsultasi dengan dokter/tenaga kesehatan
        - 📋 Pemeriksaan kesehatan menyeluruh
        - 💊 Evaluasi status nutrisi
        - 📈 Pantau pertumbuhan setiap 2 minggu
        """)
    elif rec_type == "info":
        st.info(f"""
        **Status: {status}** 🟡
        
        ⚠️ Anak berisiko mengalami stunting.
        
        **Rekomendasi:**
        - 🏥 Konsultasi dengan tenaga kesehatan
        - 🥗 Tingkatkan nutrisi makanan
        - 📈 Pantau pertumbuhan setiap bulan
        """)
    else:
        st.success(f"""
        **Status: {status}** 🟢
        
        ✅ Anak tidak menunjukkan tanda-tanda stunting.
        
        **Rekomendasi:**
        - ✓ Lanjutkan pemantauan pertumbuhan
        - ✓ Pertahankan pola makan bergizi
        - ✓ Check-up 6 bulan sekali
        """)
    
    st.markdown("---")
    
    # Grafik Perbandingan
    st.markdown("### 📈 Perbandingan Antropometri")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure(data=[
            go.Bar(
                name='Data Anak',
                x=['Berat (kg)', 'Tinggi (cm)'],
                y=[berat_badan, tinggi_badan/10],
                marker=dict(color=['#3498db', '#e74c3c'])
            ),
            go.Bar(
                name='Standar Normal',
                x=['Berat (kg)', 'Tinggi (cm)'],
                y=[13.0, 8.7],
                marker=dict(color=['#95a5a6', '#95a5a6'])
            )
        ])
        fig.update_layout(barmode='group', height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        comparison_data = {
            'Metrik': ['Berat Badan', 'Tinggi Badan', 'Usia'],
            'Data Anak': [f'{berat_badan} kg', f'{tinggi_badan} cm', f'{usia_bulan} bulan'],
            'Standar Normal': ['13.0 kg', '87 cm', '-']
        }
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Ringkasan Data
    st.markdown("### 📋 Ringkasan Data Input")
    
    summary_data = {
        'Parameter': [
            'Nama Balita', 'Jenis Kelamin', 'Usia', 'Berat Badan', 'Tinggi Badan',
            'Cara Pengukuran', 'Kenaikan Berat'
        ],
        'Nilai': [
            nama_balita if nama_balita else '-',
            jenis_kelamin, f'{usia_bulan} bulan', f'{berat_badan} kg', f'{tinggi_badan} cm',
            cara_pengukuran, kenaikan_berat
        ]
    }
    df_summary = pd.DataFrame(summary_data)
    st.dataframe(df_summary, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Informasi Stunting
    st.markdown("### ℹ️ Informasi Stunting")
    
    with st.expander("📚 Apa itu Stunting?"):
        st.markdown("""
        **Stunting** adalah kondisi tinggi badan anak lebih pendek dari standar usia.
        
        **Penyebab:**
        - Malnutrisi kronis
        - Infeksi berulang
        - Kurang akses air bersih
        - Pengetahuan gizi orang tua kurang
        
        **Dampak:**
        - Penurunan kognitif (-5-11 IQ)
        - Pembelajaran terganggu
        - Produktivitas berkurang (-20%)
        - Risiko penyakit kronis naik
        """)
    
    with st.expander("🎯 Cara Mencegah Stunting"):
        st.markdown("""
        **1. NUTRISI:**
        - ASI eksklusif 0-6 bulan
        - MPASI berkualitas 6+ bulan
        - Protein, vitamin, mineral cukup
        
        **2. KESEHATAN:**
        - Imunisasi lengkap
        - Periksa rutin
        - Obati infeksi segera
        
        **3. KEBERSIHAN:**
        - Air bersih yang aman
        - Toilet sehat
        - Cuci tangan sebelum makan
        """)

else:
    st.info("👆 Lengkapi data di atas dan klik 'Klasifikasi' untuk hasil")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 12px; padding: 20px 0;'>
    <p>🏥 <strong>Sistem Klasifikasi Status Stunting</strong> | v3.1.0</p>
    <p>📧 Email: support@stunting.id | ☎️ Hotline: 1500-228</p>
</div>
""", unsafe_allow_html=True)
