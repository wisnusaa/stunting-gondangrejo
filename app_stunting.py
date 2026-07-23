"""
Aplikasi Klasifikasi Stunting — Puskesmas Gondangrejo
Model: Random Forest + SMOTE
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import re
from io import BytesIO

# ── Konfigurasi halaman ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Deteksi Stunting | Puskesmas Gondangrejo",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS kustom ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Font & warna dasar */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #F0F7F4;
    border-right: 1px solid #C8E6D9;
}

/* Header halaman utama */
.page-header {
    background: linear-gradient(135deg, #0F6E56 0%, #1A9E7A 100%);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 24px;
    color: white;
}
.page-header h1 {
    font-size: 26px;
    font-weight: 700;
    margin: 0 0 6px;
    color: white;
}
.page-header p {
    font-size: 14px;
    opacity: 0.85;
    margin: 0;
}

/* Card hasil prediksi */
.result-card {
    border-radius: 16px;
    padding: 28px 32px;
    text-align: center;
    margin-bottom: 16px;
    border: 2px solid;
}
.result-stunting {
    background: #FEF2F0;
    border-color: #E5452A;
    color: #7A1E0C;
}
.result-normal {
    background: #EDFBF4;
    border-color: #19A05A;
    color: #0C4D2A;
}
.result-card h2 {
    font-size: 32px;
    font-weight: 700;
    margin: 10px 0 4px;
}
.result-card .result-icon {
    font-size: 48px;
    margin-bottom: 4px;
}
.result-card .result-sub {
    font-size: 14px;
    opacity: 0.7;
}

/* Metric box */
.metric-row {
    display: flex;
    gap: 12px;
    margin-top: 12px;
}
.metric-box {
    flex: 1;
    background: #F8FBFA;
    border: 1px solid #D4EDE4;
    border-radius: 10px;
    padding: 14px 16px;
    text-align: center;
}
.metric-box .m-val {
    font-size: 22px;
    font-weight: 700;
    color: #0F6E56;
}
.metric-box .m-lab {
    font-size: 11px;
    color: #6B8F7C;
    margin-top: 2px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Tabel prediksi batch */
.stDataFrame {
    border-radius: 10px !important;
    overflow: hidden;
}

/* Input section divider */
.section-label {
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #4A7A64;
    margin: 20px 0 8px;
    padding-bottom: 6px;
    border-bottom: 1px solid #C8E6D9;
}

/* Alert */
.alert-info {
    background: #E8F4FD;
    border-left: 4px solid #2196F3;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 13px;
    color: #0D47A1;
    margin: 12px 0;
}
.alert-warn {
    background: #FFF8E1;
    border-left: 4px solid #FFA000;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 13px;
    color: #5D3B00;
    margin: 12px 0;
}

/* Tombol prediksi */
div.stButton > button[kind="primary"] {
    background: #0F6E56;
    border: none;
    border-radius: 10px;
    font-size: 15px;
    font-weight: 600;
    padding: 12px 32px;
    width: 100%;
    color: white;
    transition: background .2s;
}
div.stButton > button[kind="primary"]:hover {
    background: #0A5240;
}

/* Hide default streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model():
    """Load model RF + scaler. Kalau belum ada, latih dari dataset."""
    if os.path.exists("rf_model.pkl") and os.path.exists("scaler.pkl"):
        with open("rf_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    return None, None


def encode_features(row: dict) -> np.ndarray:
    """
    Encode satu baris input ke vektor fitur sesuai urutan training.
    Menggunakan mapping kategori yang sama dengan LabelEncoder saat training.
    """
    # Mapping kategorikal (disesuaikan dengan urutan LabelEncoder fit pada data asli)
    jk_map    = {'Laki-laki': 0, 'Perempuan': 1}
    cara_map  = {'Berbaring / Terlentang': 0, 'Berdiri': 1}
    bbu_map   = {'Berat Badan Kurang': 0, 'Gizi Buruk': 1,
                 'Normal': 2, 'Risiko Gizi Lebih': 3}
    bbtb_map  = {'Gizi Baik': 0, 'Gizi Kurang': 1,
                 'Gizi Lebih': 2, 'Obesitas': 3, 'Risiko Gizi Lebih': 4}
    nbb_map   = {'Tidak Naik': 0, 'Naik': 1}

    jk_enc   = jk_map.get(row['JK'], 0)
    cara_enc = cara_map.get(row['Cara_Ukur'], 0)
    bbu_enc  = bbu_map.get(row['BBU'], 2)
    bbtb_enc = bbtb_map.get(row['BBTB'], 0)
    nbb_enc  = nbb_map.get(row['NBB'], 1)

    # Koreksi tinggi badan sesuai WHO ±0.7 cm
    tinggi = row['Tinggi']
    usia   = row['Usia_Bulan']
    cara   = row['Cara_Ukur'].lower()
    if usia > 24 and 'terlentang' in cara:
        tinggi -= 0.7
    elif usia <= 24 and 'berdiri' in cara:
        tinggi += 0.7

    return np.array([[
        jk_enc,
        usia,
        row['Berat'],
        tinggi,
        cara_enc,
        bbu_enc,
        bbtb_enc,
        nbb_enc,
    ]])


def predict_single(model, scaler, features: np.ndarray):
    X_sc = scaler.transform(features)
    pred  = model.predict(X_sc)[0]
    proba = model.predict_proba(X_sc)[0]
    return pred, proba


def interpret_result(pred, proba):
    p_stunt = proba[1] * 100
    p_norm  = proba[0] * 100
    if pred == 1:
        label = "STUNTING"
        risk  = "Tinggi" if p_stunt >= 70 else "Sedang"
    else:
        label = "NORMAL"
        risk  = "Rendah" if p_norm >= 70 else "Sedang"
    return label, risk, p_stunt, p_norm


# ══════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🏥 Deteksi Stunting")
    st.markdown("**Puskesmas Gondangrejo**")
    st.markdown("---")

    menu = st.radio(
        "Menu",
        ["📋 Prediksi Satu Balita", "📂 Prediksi Batch (File)", "📊 Info Model"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:12px;color:#4A7A64;line-height:1.7'>
    <b>Model:</b> Random Forest<br>
    <b>Teknik:</b> SMOTE (k=5)<br>
    <b>Fitur input:</b> 8 variabel<br>
    <b>Data latih:</b> 7.012 sampel<br>
    <b>Akurasi:</b> 96,84%<br>
    <b>F1-Score Stunting:</b> 0,4242
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px;color:#8AB09A'>
    ⚠️ Hasil prediksi ini merupakan alat bantu keputusan klinis, 
    bukan diagnosis final. Keputusan tetap pada tenaga medis.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# Load model
# ══════════════════════════════════════════════════════════════════════════

model, scaler = load_model()

# ══════════════════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="page-header">
    <h1>🩺 Sistem Deteksi Dini Stunting</h1>
    <p>Berbasis Machine Learning — Random Forest + SMOTE &nbsp;|&nbsp; 
    Dataset Puskesmas Gondangrejo &nbsp;|&nbsp; 
    Universitas Amikom Yogyakarta</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 1 — Prediksi Satu Balita
# ══════════════════════════════════════════════════════════════════════════

if menu == "📋 Prediksi Satu Balita":

    if model is None:
        st.markdown("""
        <div class="alert-warn">
        ⚠️ <b>File model belum ditemukan.</b> Silakan latih model terlebih dahulu 
        menggunakan notebook <code>MODEL_KLASIFIKASI_RF_SMOTE.ipynb</code> dan simpan 
        hasilnya sebagai <code>rf_model.pkl</code> dan <code>scaler.pkl</code> 
        di folder yang sama dengan aplikasi ini.
        </div>
        """, unsafe_allow_html=True)

        with st.expander("📌 Cara menyimpan model dari notebook"):
            st.code("""
import pickle

# Simpan model Random Forest
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(rf_smote, f)

# Simpan scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler_sm, f)

print("✅ Model berhasil disimpan")
            """, language="python")
        st.stop()

    col_input, col_result = st.columns([1.1, 0.9], gap="large")

    with col_input:
        st.markdown("#### Data Pengukuran Balita")

        st.markdown('<div class="section-label">Identitas</div>', unsafe_allow_html=True)
        nama_balita = st.text_input("Nama Balita (opsional)", placeholder="Contoh: Budi Santoso")

        col_a, col_b = st.columns(2)
        with col_a:
            jk = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
        with col_b:
            usia = st.number_input("Usia (bulan)", min_value=0, max_value=59,
                                   value=24, step=1,
                                   help="Rentang valid: 0–59 bulan (sesuai standar WHO)")

        st.markdown('<div class="section-label">Antropometri</div>', unsafe_allow_html=True)

        col_c, col_d = st.columns(2)
        with col_c:
            berat = st.number_input("Berat Badan (kg)", min_value=1.0, max_value=30.0,
                                    value=10.5, step=0.1, format="%.1f")
        with col_d:
            tinggi = st.number_input("Tinggi / Panjang Badan (cm)",
                                     min_value=30.0, max_value=130.0,
                                     value=80.0, step=0.1, format="%.1f")

        cara_ukur = st.selectbox(
            "Cara Pengukuran",
            ["Berbaring / Terlentang", "Berdiri"],
            help="WHO: usia ≤24 bln sebaiknya terlentang; >24 bln berdiri"
        )

        # Koreksi otomatis
        tinggi_koreksi = tinggi
        if usia > 24 and "terlentang" in cara_ukur.lower():
            tinggi_koreksi = tinggi - 0.7
            st.info(f"📏 Koreksi WHO diterapkan: {tinggi} − 0,7 = **{tinggi_koreksi:.1f} cm**")
        elif usia <= 24 and "berdiri" in cara_ukur.lower():
            tinggi_koreksi = tinggi + 0.7
            st.info(f"📏 Koreksi WHO diterapkan: {tinggi} + 0,7 = **{tinggi_koreksi:.1f} cm**")

        st.markdown('<div class="section-label">Status Gizi</div>', unsafe_allow_html=True)

        col_e, col_f = st.columns(2)
        with col_e:
            bbu = st.selectbox("Status BB/U", [
                "Normal", "Risiko Gizi Lebih",
                "Berat Badan Kurang", "Gizi Buruk"
            ])
        with col_f:
            bbtb = st.selectbox("Status BB/TB", [
                "Gizi Baik", "Risiko Gizi Lebih",
                "Gizi Lebih", "Gizi Kurang", "Obesitas"
            ])

        nbb = st.selectbox("Kenaikan Berat Badan dari Pengukuran Sebelumnya",
                           ["Naik", "Tidak Naik"])

        st.markdown("")
        btn_predict = st.button("🔍 Prediksi Status Stunting", type="primary")

    with col_result:
        st.markdown("#### Hasil Prediksi")

        if btn_predict:
            row = {
                'JK': jk, 'Usia_Bulan': usia, 'Berat': berat,
                'Tinggi': tinggi, 'Cara_Ukur': cara_ukur,
                'BBU': bbu, 'BBTB': bbtb, 'NBB': nbb,
            }
            features = encode_features(row)
            pred, proba = predict_single(model, scaler, features)
            label, risk, p_stunt, p_norm = interpret_result(pred, proba)

            if pred == 1:
                st.markdown(f"""
                <div class="result-card result-stunting">
                    <div class="result-icon">⚠️</div>
                    <h2>{label}</h2>
                    <div class="result-sub">Risiko: {risk}</div>
                </div>
                """, unsafe_allow_html=True)
                st.error(
                    "**Tindak lanjut disarankan:** Segera lakukan pemeriksaan "
                    "lanjutan oleh dokter atau ahli gizi. Konsultasikan intervensi "
                    "gizi sesuai protokol penanganan stunting."
                )
            else:
                st.markdown(f"""
                <div class="result-card result-normal">
                    <div class="result-icon">✅</div>
                    <h2>{label}</h2>
                    <div class="result-sub">Risiko: {risk}</div>
                </div>
                """, unsafe_allow_html=True)
                st.success(
                    "Status pertumbuhan baik. Tetap lakukan pemantauan rutin "
                    "setiap bulan dan pastikan asupan gizi sesuai usia."
                )

            # Probabilitas
            st.markdown("**Probabilitas Prediksi**")
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.metric("Normal", f"{p_norm:.1f}%")
            with col_p2:
                st.metric("Stunting", f"{p_stunt:.1f}%")

            st.progress(p_stunt / 100)
            st.caption(f"Semakin tinggi nilai di atas, semakin besar risiko stunting")

            # Detail input
            with st.expander("📋 Ringkasan Data Inputan"):
                nama_display = nama_balita if nama_balita else "—"
                st.markdown(f"""
                | Parameter | Nilai |
                |-----------|-------|
                | Nama | {nama_display} |
                | Jenis Kelamin | {jk} |
                | Usia | {usia} bulan |
                | Berat Badan | {berat} kg |
                | Tinggi Badan | {tinggi} cm |
                | Tinggi Terkoreksi | {tinggi_koreksi:.1f} cm |
                | Cara Ukur | {cara_ukur} |
                | Status BB/U | {bbu} |
                | Status BB/TB | {bbtb} |
                | Naik Berat Badan | {nbb} |
                """)

        else:
            st.markdown("""
            <div class="alert-info">
            ℹ️ Isi data pengukuran balita di sebelah kiri, 
            lalu klik tombol <b>Prediksi Status Stunting</b>.
            </div>
            """, unsafe_allow_html=True)

            # Panduan input
            with st.expander("❓ Panduan Pengisian Data"):
                st.markdown("""
                **Usia:** Hitung dari tanggal lahir hingga tanggal pengukuran dalam bulan penuh.

                **Cara Pengukuran:**
                - Anak ≤ 24 bulan → sebaiknya **Terlentang** (diukur panjang badan)
                - Anak > 24 bulan → sebaiknya **Berdiri** (diukur tinggi badan)
                - Jika tidak sesuai, koreksi ±0,7 cm akan diterapkan otomatis.

                **Status BB/U:** Hasil dari timbangan berat badan dibandingkan standar WHO menurut usia.

                **Status BB/TB:** Hasil dari perbandingan berat badan terhadap tinggi badan.

                **Naik Berat Badan:** Apakah berat badan anak naik dibandingkan pengukuran bulan sebelumnya?
                """)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 2 — Prediksi Batch
# ══════════════════════════════════════════════════════════════════════════

elif menu == "📂 Prediksi Batch (File)":

    if model is None:
        st.warning("⚠️ File model belum ditemukan. Lihat menu Prediksi Satu Balita untuk instruksi.")
        st.stop()

    st.markdown("#### Prediksi Massal dari File Excel / CSV")
    st.markdown("""
    <div class="alert-info">
    ℹ️ Upload file dengan kolom: <b>Nama, JK, Usia_Bulan, Berat, Tinggi, Cara_Ukur, BBU, BBTB, NBB</b>
    </div>
    """, unsafe_allow_html=True)

    # Template download
    template_data = pd.DataFrame({
        'Nama':       ['Budi S', 'Ani W'],
        'JK':         ['Laki-laki', 'Perempuan'],
        'Usia_Bulan': [18, 30],
        'Berat':      [9.5, 12.0],
        'Tinggi':     [75.0, 88.0],
        'Cara_Ukur':  ['Berbaring / Terlentang', 'Berdiri'],
        'BBU':        ['Normal', 'Normal'],
        'BBTB':       ['Gizi Baik', 'Gizi Baik'],
        'NBB':        ['Naik', 'Tidak Naik'],
    })

    buf = BytesIO()
    template_data.to_excel(buf, index=False)
    st.download_button(
        "⬇️ Download Template Excel",
        data=buf.getvalue(),
        file_name="template_prediksi_stunting.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    uploaded = st.file_uploader("Upload file data balita", type=["xlsx", "csv"])

    if uploaded:
        try:
            if uploaded.name.endswith(".csv"):
                df_up = pd.read_csv(uploaded)
            else:
                df_up = pd.read_excel(uploaded)

            st.markdown(f"**{len(df_up)} data berhasil dimuat**")

            results = []
            for _, row in df_up.iterrows():
                inp = {
                    'JK':        str(row.get('JK', 'Laki-laki')),
                    'Usia_Bulan': int(row.get('Usia_Bulan', 0)),
                    'Berat':      float(row.get('Berat', 0)),
                    'Tinggi':     float(row.get('Tinggi', 0)),
                    'Cara_Ukur':  str(row.get('Cara_Ukur', 'Berdiri')),
                    'BBU':        str(row.get('BBU', 'Normal')),
                    'BBTB':       str(row.get('BBTB', 'Gizi Baik')),
                    'NBB':        str(row.get('NBB', 'Naik')),
                }
                feat = encode_features(inp)
                pred, proba = predict_single(model, scaler, feat)
                results.append({
                    'Nama':         row.get('Nama', '—'),
                    'Usia (bln)':   inp['Usia_Bulan'],
                    'Berat (kg)':   inp['Berat'],
                    'Tinggi (cm)':  inp['Tinggi'],
                    'Prediksi':     '⚠️ STUNTING' if pred == 1 else '✅ NORMAL',
                    'P(Stunting)':  f"{proba[1]*100:.1f}%",
                    'P(Normal)':    f"{proba[0]*100:.1f}%",
                })

            df_result = pd.DataFrame(results)

            # Ringkasan
            n_stunting = sum(1 for r in results if 'STUNTING' in r['Prediksi'])
            n_normal   = len(results) - n_stunting

            col_s1, col_s2, col_s3 = st.columns(3)
            col_s1.metric("Total Balita", len(results))
            col_s2.metric("Terindikasi Stunting", n_stunting,
                          delta=f"{n_stunting/len(results)*100:.1f}%",
                          delta_color="inverse")
            col_s3.metric("Normal", n_normal)

            st.dataframe(df_result, use_container_width=True, height=400)

            # Download hasil
            buf_out = BytesIO()
            df_result.to_excel(buf_out, index=False)
            st.download_button(
                "⬇️ Download Hasil Prediksi (.xlsx)",
                data=buf_out.getvalue(),
                file_name="hasil_prediksi_stunting.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        except Exception as e:
            st.error(f"Gagal memproses file: {e}")
            st.info("Pastikan nama kolom sesuai dengan template yang tersedia.")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 3 — Info Model
# ══════════════════════════════════════════════════════════════════════════

elif menu == "📊 Info Model":

    st.markdown("#### Informasi Model Klasifikasi")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 🤖 Spesifikasi Model")
        st.markdown("""
        | Parameter | Nilai |
        |-----------|-------|
        | Algoritma | Random Forest Classifier |
        | n_estimators | 100 pohon |
        | criterion | Gini Impurity |
        | random_state | 42 |
        | Teknik balancing | SMOTE (k_neighbors=5) |
        | Normalisasi | MinMaxScaler |
        """)

        st.markdown("##### 📦 Dataset")
        st.markdown("""
        | Parameter | Nilai |
        |-----------|-------|
        | Sumber | Puskesmas Gondangrejo |
        | Total sampel | 4.816 |
        | Rentang usia | 0–59 bulan |
        | Kelas Stunting | 141 (2,9%) |
        | Kelas Normal | 4.675 (97,1%) |
        | Rasio imbalance | 33 : 1 |
        | Data latih | 3.612 (75%) |
        | Data uji | 1.204 (25%) |
        | Setelah SMOTE | 7.012 (seimbang) |
        """)

    with col2:
        st.markdown("##### 📈 Performa Model (Data Uji)")
        st.markdown("""
        | Metrik | Nilai |
        |--------|-------|
        | Accuracy | 96,84% |
        | Precision (Stunting) | 45,16% |
        | Recall (Stunting) | 40,00% |
        | F1-Score (Stunting) | 0,4242 |
        | CV F1-Score | 0,5105 ± 0,0953 |
        | TP | 14 |
        | TN | 1.152 |
        | FP | 17 |
        | FN | 21 |
        """)

        st.markdown("##### 🏆 Perbandingan Algoritma")
        df_comp = pd.DataFrame({
            'Algoritma':   ['Random Forest', 'SVM', 'KNN', 'Naive Bayes', 'Decision Tree'],
            'Accuracy':    ['96,84%', '91,78%', '93,52%', '71,10%', '63,37%'],
            'Precision':   ['45,16%', '22,88%', '22,08%', '7,12%',  '5,48%'],
            'Recall':      ['40,00%', '77,14%', '48,57%', '74,29%', '71,43%'],
            'F1-Score':    ['0,4242', '0,3529', '0,3036', '0,1300', '0,1018'],
        })
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("##### 📌 Fitur Input yang Digunakan")
    df_feat = pd.DataFrame({
        'No': range(1, 9),
        'Fitur': ['JK_Enc', 'Usia_Bulan', 'Berat', 'Tinggi_Koreksi',
                  'Cara_Enc', 'BBU_Enc', 'BBTB_Enc', 'NBB_Enc'],
        'Deskripsi': [
            'Jenis kelamin (dikodekan: L=0, P=1)',
            'Usia dalam bulan (0–59)',
            'Berat badan saat pengukuran (kg)',
            'Tinggi badan terkoreksi ±0,7 cm sesuai standar WHO (cm)',
            'Cara pengukuran: Terlentang=0, Berdiri=1',
            'Status berat badan per umur (dikodekan LabelEncoder)',
            'Status berat badan per tinggi (dikodekan LabelEncoder)',
            'Status kenaikan berat badan: Tidak Naik=0, Naik=1',
        ],
        'Tipe': ['Kategorikal', 'Numerik', 'Numerik', 'Numerik',
                 'Kategorikal', 'Kategorikal', 'Kategorikal', 'Kategorikal'],
    })
    st.dataframe(df_feat, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("""
    <div class="alert-warn">
    ⚠️ <b>Perhatian:</b> Kolom <code>TB/U</code> dan <code>ZS TB/U</code> 
    <b>tidak</b> digunakan sebagai fitur input untuk menghindari <i>data leakage</i>. 
    Kolom TB/U hanya digunakan sebagai sumber label target saat pelatihan.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("##### 📚 Referensi Utama")
    st.markdown("""
    - Breiman, L. (2001). **Random Forests**. *Machine Learning*, 45(1), 5–32.
    - Chawla et al. (2002). **SMOTE: Synthetic Minority Over-sampling Technique**. *JAIR*, 16, 321–357.
    - Kemenkes RI (2023). **Buku Saku SSGI 2022**. Badan Kebijakan Pembangunan Kesehatan.
    - Satria et al. (2024). **Perbaikan Akurasi Random Forest dengan ANOVA dan SMOTE pada Data Stunting**. *Teknika*.
    """)
