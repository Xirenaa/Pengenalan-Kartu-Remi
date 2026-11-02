"""Aplikasi web Streamlit: Pengenalan Kartu Remi (Kelompok 9, UTS Praktikum Visi Komputer).

Jalankan lokal:  streamlit run app.py
"""
import csv
import os

import cv2
import streamlit as st

import kartu_remi as kr

st.set_page_config(page_title="Pengenalan Kartu Remi", page_icon="🃏", layout="wide")


@st.cache_resource(show_spinner="Memuat 12 template dan menghitung fitur ORB/SIFT...")
def muat_db():
    return kr.siapkan_database()


@st.cache_data
def daftar_contoh():
    # foto uji dari dataset beserta jawaban benarnya (ground_truth.csv)
    with open(os.path.join(kr.DATASET_DIR, "ground_truth.csv"), newline="", encoding="utf-8-sig") as f:
        return [(r["file"], r["kartu"]) for r in csv.DictReader(f)]


def rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


DB = muat_db()

# ---------------------------- Sidebar ----------------------------
with st.sidebar:
    st.header("Pengaturan")
    pilihan = st.radio(
        "Algoritma fitur",
        ["SIFT", "ORB"],
        help="SIFT lebih akurat pada data kami (67,9% vs 53,6%), ORB sekitar 4x lebih cepat.",
    )
    metode = pilihan.lower()
    st.caption(f"Ambang \"tidak dikenali\": skor good match < {kr.AMBANG[kr.SKOR_MODE][metode]}")
    st.divider()
    st.markdown(
        "**Kelompok 9 — Tema 9**  \n"
        "Muhammad Yazid Arrazi (2408107010072)  \n"
        "Muhammad Farhan Alafif (2408107010075)  \n"
        "Muhammad Razi Siregar (2408107010101)"
    )
    st.caption("UTS Praktikum Visi Komputer, Informatika USK")

# ---------------------------- Halaman utama ----------------------------
st.title("🃏 Pengenalan Kartu Remi")
st.write(
    "Unggah foto kartu remi (nilai **3, 7, atau 10** dari jenis hati, wajik, keriting, sekop). "
    "Program mendeteksi kartu, meluruskannya, lalu mencocokkan fitur **SIFT/ORB** dengan 12 template. "
    "Satu foto boleh berisi **maksimal dua kartu**."
)

sumber = st.radio("Sumber gambar", ["Unggah foto", "Contoh dari dataset"], horizontal=True)

bytes_gambar, nama_file, kunci = None, None, None
if sumber == "Unggah foto":
    f = st.file_uploader("Pilih foto (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if f is not None:
        bytes_gambar, nama_file = f.getvalue(), f.name
else:
    contoh = dict(daftar_contoh())
    pilih = st.selectbox("Pilih foto uji", list(contoh), format_func=lambda x: f"{x}  (jawaban: {contoh[x].replace('_', ' ')})")
    with open(os.path.join(kr.DATASET_DIR, pilih), "rb") as fh:
        bytes_gambar, nama_file, kunci = fh.read(), pilih, contoh[pilih]

if bytes_gambar is None:
    st.info("Belum ada gambar. Unggah foto kartu remi di atas untuk memulai.")
    st.stop()

try:
    with st.spinner("Memproses..."):
        asli, hasil_img, hasil, waktu = kr.proses_foto(bytes_gambar, metode, db=DB)
except Exception as e:  # file rusak / bukan gambar
    st.error(f"Gambar tidak bisa diproses: {e}")
    st.stop()

# gambar asli dan hasil berdampingan
k1, k2 = st.columns(2)
with k1:
    st.subheader("Gambar asli")
    st.image(rgb(asli), width="stretch")
with k2:
    st.subheader(f"Hasil pengenalan ({pilihan})")
    st.image(rgb(hasil_img), width="stretch")
    st.caption("Kotak hijau = kartu dikenali (nama + jumlah good match). Kotak merah = skor di bawah ambang, tidak dikenali.")

# informasi hasil
dikenali = [h for h in hasil if h["diterima"]]
st.subheader("Informasi hasil")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Kartu dikenali", len(dikenali))
m2.metric("Kandidat kartu terdeteksi", len(hasil))
m3.metric("Waktu proses", f"{waktu * 1000:.0f} ms")
m4.metric("Algoritma", pilihan)

if not hasil:
    st.warning("Tidak ada kartu terdeteksi. Coba foto dari atas, kartu lebih besar di bingkai, dan latar polos tanpa pantulan lampu.")
elif not dikenali:
    st.warning("Kartu terdeteksi tetapi skor kecocokan di bawah ambang, jadi hasilnya \"tidak dikenali\".")
else:
    st.success("Label: " + ", ".join(kr.nama_tampil(h["nama"]) for h in dikenali))

if kunci is not None:
    st.caption(f"Jawaban benar (ground truth) untuk {nama_file}: " + ", ".join(x.replace('_', ' ').title() for x in kunci.split('|')))

if hasil:
    st.dataframe(
        [
            {
                "Kartu": i,
                "Label": kr.nama_tampil(h["nama"]) if h["diterima"] else "tidak dikenali",
                "Template terbaik": kr.nama_tampil(h["nama"]),
                "Good match": h["good"],
                "Inlier RANSAC": h["inlier"],
                "Deteksi": "cadangan boundingRect" if h["cadangan"] else "kontur 4 sudut",
                "Status": "dikenali" if h["diterima"] else "di bawah ambang",
            }
            for i, h in enumerate(hasil, 1)
        ],
        hide_index=True,
        width="stretch",
    )

    with st.expander("Lihat kartu hasil pelurusan dan garis matching"):
        for i, h in enumerate(hasil, 1):
            c1, c2 = st.columns([1, 3])
            c1.image(rgb(h["warp"]), caption=f"Kartu {i} setelah warpPerspective", width="stretch")
            gm = kr.gambar_match(h)
            if gm is not None:
                c2.image(rgb(gm), caption=f"Matching dengan template {kr.nama_tampil(h['nama'])} (50 padanan terbaik)", width="stretch")
