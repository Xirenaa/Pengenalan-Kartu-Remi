"""Pipeline pengenalan kartu remi (Kelompok 9, UTS Praktikum Visi Komputer).

Kode ini diambil dari notebook UTS_Kelompok09.ipynb (bagian 2) tanpa mengubah logika pipeline.
Yang dibuang hanya bagian khusus notebook (tampilan matplotlib, pencarian dataset Kaggle, loop evaluasi).
Dipakai oleh app.py (Streamlit).
"""
import os
import time
from collections import OrderedDict

import numpy as np
import cv2

cv2.ocl.setUseOpenCL(False)      # sama seperti modul Prak 5

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
TEMPLATE_DIR = os.path.join(DATASET_DIR, 'template')

# ===================== PARAMETER (ubah di sini) =====================
SISI_MAKS = 1000               # sisi terpanjang foto setelah resize (soal: 800-1200 px)
UKURAN_KARTU = (300, 420)      # (lebar, tinggi) hasil warp; kartu poker 63x88 mm -> rasio 1,4

BLUR_K = (7, 7)                # kernel Gaussian blur (nilai di modul Prak 3)
CANNY = (40, 120)              # ambang bawah/atas Canny (rasio 1:3)
ADAPT_BLOK, ADAPT_C = 201, -8  # adaptive threshold: blok > ukuran kartu; C negatif = ambil yang lebih terang dari rata-rata lokal
STRATEGI = ('otsu', 'adaptif', 'canny')   # strategi segmentasi; kandidat dari semuanya digabung
MIN_AREA, MAX_AREA = 0.01, 0.60           # luas kartu / luas foto
EPS_LIST = (0.02, 0.03, 0.04, 0.05, 0.06) # epsilon approxPolyDP (x keliling), dicoba bertahap
RASIO_KARTU = 1.4              # tinggi/lebar kartu poker
MAKS_KANDIDAT = 3              # maksimum kandidat kartu per foto
SUDUT_TOL = 25                 # toleransi penyimpangan sudut segi empat dari 90 derajat (pantulan lampu membuat segi empat miring)
RASIO_LUAS_MIN = 0.35          # kandidat lebih kecil dari 35% kandidat terbesar dibuang (bayangan/noise)
MAX_AREA_TEMPLATE = 0.50       # template: kartu tidak mungkin memenuhi seluruh foto (mencegah seluruh frame dianggap kartu)

SIFT_NFEATURES = 1000          # batas keypoint SIFT per citra (default tanpa batas: ribuan keypoint, sangat lambat)
RASIO = 0.75                   # ratio test Lowe (modul Prak 5 memakai 0,75)
RANSAC_THR = 4                 # reprojThresh findHomography (modul Prak 5 memakai 4)
ORIENTASI = (0, 180)           # sudut putar yang dicoba pada kartu hasil warp (Prak 3)
PAKAI_EQ = True                # histogram equalization sebelum ekstraksi fitur (Prak 3)
SKOR_MODE = 'good'             # 'good' = jumlah good match; 'inlier' = good match yang lolos RANSAC
# skor minimum agar kartu dianggap dikenali, per mode skor dan per detektor (dasar: histogram skor di bagian 5).
# SIFT: pada run foto asli, kartu sungguhan >= 23 good match dan kandidat palsu <= 12, sehingga ambang 20 memisahkan keduanya.
# ORB: skor lebih kecil, ambang 10 dipertahankan.
AMBANG = {'good': {'sift': 20, 'orb': 10}, 'inlier': {'sift': 8, 'orb': 8}}
FILTER_WARNA = True            # True = batasi template ke warna yang sama (merah/hitam) lewat HSV (Prak 3); ketepatan warna dicek di bagian 6
DETEKTOR_UTAMA = 'sift'        # detektor untuk hasil utama; ubah setelah melihat bagian 6
DAFTAR_DETEKTOR = ('orb', 'sift')
KONDISI_ROTASI = ('rotasi', 'ideal')            # kondisi yang masuk tabel per sudut
KONDISI_SKALA = ('skala', 'rotasi', 'ideal')    # kondisi yang masuk tabel per jarak



# ===================== FUNGSI BANTU UMUM (Prak 1-3) =====================
def baca_citra(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f'Gambar tidak bisa dibaca: {path}')
    return img


def resize_maks(img, sisi_maks=SISI_MAKS):
    # cv2.resize dengan faktor skala dan interpolasi (Prak 3)
    s = sisi_maks / max(img.shape[:2])
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, None, fx=s, fy=s, interpolation=interp)


def putar(img, sudut):
    # rotasi terhadap pusat citra: getRotationMatrix2D + warpAffine (Prak 3)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), sudut, 1)
    return cv2.warpAffine(img, M, (w, h))


def nama_tampil(nama):
    return nama.replace('_', ' ').title() if nama else 'tidak dikenali'




# ===================== KODE DARI MODUL PRAK 5 (fitur dan matching) =====================
def detectAndDescribe(image, method=None):
    """Hitung keypoint dan descriptor dengan metode tertentu (kode modul Prak 5)."""
    assert method is not None, "Tentukan metode: 'sift' atau 'orb'"
    if method == 'sift':
        descriptor = cv2.SIFT_create(nfeatures=SIFT_NFEATURES)   # di modul: cv2.xfeatures2d.SIFT_create()
    elif method == 'orb':
        descriptor = cv2.ORB_create()
    (kps, features) = descriptor.detectAndCompute(image, None)
    return (kps, features)


def createMatcher(method, crossCheck):
    "Create and return a Matcher Object (kode modul Prak 5)"
    if method == 'sift' or method == 'surf':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
    elif method == 'orb' or method == 'brisk':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)
    return bf


def matchKeyPointsKNN(featuresA, featuresB, ratio, method):
    # kNN k=2 + Lowe's ratio test (kode modul Prak 5; print dihapus, ditambah pengaman len(pair) == 2)
    bf = createMatcher(method, crossCheck=False)
    rawMatches = bf.knnMatch(featuresA, featuresB, 2)
    matches = []
    for pair in rawMatches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < n.distance * ratio:
            matches.append(m)
    return matches


def getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh):
    # homografi RANSAC dari padanan fitur (kode modul Prak 5)
    kpsA = np.float32([kp.pt for kp in kpsA])
    kpsB = np.float32([kp.pt for kp in kpsB])
    if len(matches) > 4:
        ptsA = np.float32([kpsA[m.queryIdx] for m in matches])
        ptsB = np.float32([kpsB[m.trainIdx] for m in matches])
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
        return (matches, H, status)
    else:
        return None


# ===================== TAHAP 1: PRAPROSES DAN SEGMENTASI =====================
def praproses(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     # Prak 2/3
    blur = cv2.GaussianBlur(gray, BLUR_K, 0)         # Prak 3
    return gray, blur


def segmentasi(blur, strategi):
    if strategi == 'otsu':       # Prak 3: Otsu; kartu putih -> piksel terang jadi foreground
        _, m = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        return cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    if strategi == 'adaptif':    # Prak 3: adaptive thresholding (rata-rata Gaussian)
        m = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, ADAPT_BLOK, ADAPT_C)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        return cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    if strategi == 'canny':      # Prak 4: tepi Canny, ditebalkan lalu ditutup agar kontur tertutup
        tepi = cv2.Canny(blur, CANNY[0], CANNY[1])
        tepi = cv2.dilate(tepi, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
        return cv2.morphologyEx(tepi, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    raise ValueError(f'strategi tidak dikenal: {strategi}')


def cari_kontur(mask):
    # findContours seperti di modul Prak 5; imutils.grab_contours diganti satu baris setara
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cnts[0] if len(cnts) == 2 else cnts[1]


# ===================== TAHAP 2: KONTUR -> 4 SUDUT KARTU (Prak 4) =====================
def kontur_ke_quad(kontur):
    hull = cv2.convexHull(kontur)
    luas_hull = cv2.contourArea(hull)
    if luas_hull <= 0:
        return None, 0
    keliling = cv2.arcLength(hull, True)
    quad = None
    for eps in EPS_LIST:                        # naikkan epsilon sampai tepat 4 titik
        approx = cv2.approxPolyDP(hull, eps * keliling, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype(np.float32)
            break
    if quad is None:                            # cadangan: persegi panjang terkecil
        rect = cv2.minAreaRect(hull)
        w, h = rect[1]
        if w * h > 0 and luas_hull / (w * h) > 0.85:
            quad = cv2.boxPoints(rect).astype(np.float32)
    if quad is None:
        return None, luas_hull
    rasio_luas = luas_hull / max(cv2.contourArea(quad), 1e-6)
    if not (0.9 <= rasio_luas <= 1.15):         # bentuk hull harus mirip segi empat hasil aproksimasi
        return None, luas_hull
    return quad, luas_hull


def urutkan_titik(pts):
    # Urut searah jarum jam dari pusat, sisi PENDEK di atas (aman untuk kartu 45 derajat).
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    pusat = pts.mean(axis=0)
    sudut = np.arctan2(pts[:, 1] - pusat[1], pts[:, 0] - pusat[0])
    p = pts[np.argsort(sudut)]
    sisi = [np.linalg.norm(p[(i + 1) % 4] - p[i]) for i in range(4)]
    if sisi[0] + sisi[2] > sisi[1] + sisi[3]:
        p = np.roll(p, -1, axis=0)
    return p


def ukuran_quad(p):
    lebar = (np.linalg.norm(p[1] - p[0]) + np.linalg.norm(p[2] - p[3])) / 2
    tinggi = (np.linalg.norm(p[3] - p[0]) + np.linalg.norm(p[2] - p[1])) / 2
    return lebar, tinggi


def selisih_sudut(p):
    # Penyimpangan terbesar sudut segi empat dari 90 derajat. Kartu asli hampir persegi panjang,
    # sedangkan kontur yang menyatu dengan pantulan lampu menjadi jajar genjang miring.
    dev = 0.0
    for i in range(4):
        a, b = p[i - 1] - p[i], p[(i + 1) % 4] - p[i]
        cosv = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
        dev = max(dev, abs(np.degrees(np.arccos(np.clip(cosv, -1, 1))) - 90))
    return dev


# ===================== TAHAP 3: PELURUSAN (warpPerspective, Prak 5) =====================
def luruskan(img, quad, ukuran=UKURAN_KARTU, asli=None):
    # img = citra kerja (hasil resize, tempat kartu dideteksi). asli = foto resolusi penuh (opsional):
    # kalau ada, kartu di-warp dari foto asli supaya tajam (kartu kecil di foto 1000 px akan buram bila diperbesar).
    p = urutkan_titik(quad)
    sumber, titik = img, p
    if asli is not None:
        f = asli.shape[1] / img.shape[1]
        sumber, titik = asli, p * f
    W, H = ukuran
    W2, H2 = 2 * W, 2 * H                       # warp di 2x ukuran lalu diperkecil (INTER_AREA, Prak 3) agar tajam
    dst = np.float32([[0, 0], [W2 - 1, 0], [W2 - 1, H2 - 1], [0, H2 - 1]])
    M, _ = cv2.findHomography(titik, dst)       # 4 pasang titik -> homografi tepat
    warp = cv2.warpPerspective(sumber, M, (W2, H2))
    return cv2.resize(warp, (W, H), interpolation=cv2.INTER_AREA), p


def deteksi_kartu(img, maks_kartu=MAKS_KANDIDAT, min_area=MIN_AREA, max_area=MAX_AREA, asli=None):
    _, blur = praproses(img)
    area_img = img.shape[0] * img.shape[1]
    kandidat = []
    mask_otsu = None
    for s in STRATEGI:
        mask = segmentasi(blur, s)
        if s == 'otsu':
            mask_otsu = mask
        for c in cari_kontur(mask):
            x, y, bw, bh = cv2.boundingRect(c)
            if bw * bh < 0.5 * min_area * area_img:     # saring cepat: terlalu kecil
                continue
            quad, luas = kontur_ke_quad(c)
            if quad is None or not (min_area <= luas / area_img <= max_area):
                continue
            p = urutkan_titik(quad)
            w, h = ukuran_quad(p)
            if w < 1:
                continue
            # utamakan luas besar, rasio mirip kartu (1,4), dan sudut mendekati 90 derajat
            nilai = (luas * np.exp(-(((h / w) - RASIO_KARTU) / 0.3) ** 2)
                     * np.exp(-(selisih_sudut(p) / SUDUT_TOL) ** 2))
            kandidat.append((nilai, luas, p, s))
    kandidat.sort(key=lambda x: -x[0])
    if kandidat:                                        # buang kandidat kecil (bayangan/noise) dibanding yang terbesar
        luas_maks = max(k[1] for k in kandidat[:maks_kartu + 2])
        kandidat = [k for k in kandidat if k[1] >= RASIO_LUAS_MIN * luas_maks or k is kandidat[0]]
    terpilih = []
    for nilai, luas, p, s in kandidat:                  # buang kandidat kembar antar-strategi
        pusat = p.mean(axis=0)
        kembar = any(np.linalg.norm(pusat - t['quad'].mean(axis=0)) < 0.5 * np.sqrt(min(luas, t['luas']))
                     for t in terpilih)
        if not kembar:
            warp, p2 = luruskan(img, p, asli=asli)
            terpilih.append(dict(quad=p2, luas=luas, strategi=s, warp=warp, cadangan=False, cache={}))
        if len(terpilih) >= maks_kartu:
            break
    if not terpilih and mask_otsu is not None:          # cadangan: ROI boundingRect kontur terbesar (modul Prak 5)
        cnts = cari_kontur(mask_otsu)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            sumber, f = (img, 1.0) if asli is None else (asli, asli.shape[1] / img.shape[1])
            roi = sumber[int(y * f):int((y + h) * f), int(x * f):int((x + w) * f)]
            if roi.size > 0 and w * h >= min_area * area_img:
                if w > h:                                # buat tegak
                    roi = cv2.rotate(roi, cv2.ROTATE_90_CLOCKWISE)
                roi = cv2.resize(roi, UKURAN_KARTU, interpolation=cv2.INTER_AREA)
                quad = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
                terpilih.append(dict(quad=quad, luas=w * h, strategi='cadangan', warp=roi,
                                     cadangan=True, cache={}))
    return terpilih


def normalisasi_template(img, asli=None):
    # Template idealnya sudah di-crop. Kalau masih ada latar, kartu dicari dan diluruskan otomatis.
    # max_area dibatasi: seluruh frame foto tidak boleh dianggap sebagai kartu.
    kand = deteksi_kartu(img, maks_kartu=1, min_area=0.005, max_area=MAX_AREA_TEMPLATE, asli=asli)
    if kand and not kand[0]['cadangan']:
        return kand[0]['warp'], True
    src = img if asli is None else asli
    h, w = src.shape[:2]
    if w > h:
        src = cv2.rotate(src, cv2.ROTATE_90_CLOCKWISE)
    return cv2.resize(src, UKURAN_KARTU, interpolation=cv2.INTER_AREA), False


# ===================== TAHAP 4: PENGENALAN (ORB/SIFT + BFMatcher + ratio test) =====================
def fitur_kartu(kartu_bgr, method):
    g = cv2.cvtColor(kartu_bgr, cv2.COLOR_BGR2GRAY)
    if PAKAI_EQ:
        g = cv2.equalizeHist(g)                        # Prak 3
    return detectAndDescribe(g, method)


def warna_kartu(warp):
    # Konversi HSV (Prak 3): proporsi piksel merah pada bagian dalam kartu
    h, w = warp.shape[:2]
    isi = warp[int(0.08 * h):int(0.92 * h), int(0.08 * w):int(0.92 * w)]
    hsv = cv2.cvtColor(isi, cv2.COLOR_BGR2HSV)
    merah = ((hsv[..., 0] < 10) | (hsv[..., 0] > 165)) & (hsv[..., 1] > 90) & (hsv[..., 2] > 60)
    return 'merah' if merah.mean() > 0.004 else 'hitam'


def warna_dari_nama(nama):
    return 'merah' if nama.endswith(('hati', 'wajik')) else 'hitam'


def muat_template(template_dir=None):
    template_dir = template_dir or TEMPLATE_DIR
    daftar = OrderedDict()
    if not os.path.isdir(template_dir):
        raise FileNotFoundError(f'Folder template tidak ada: {template_dir}')
    for fn in sorted(os.listdir(template_dir)):
        stem, ext = os.path.splitext(fn)
        if ext.lower() not in ('.jpg', '.jpeg', '.png'):
            continue
        asli = baca_citra(os.path.join(template_dir, fn))
        img = resize_maks(asli)
        kartu, terdeteksi = normalisasi_template(img, asli)
        if not terdeteksi:
            print(f'[peringatan] template {fn}: kartu tidak terdeteksi, citra utuh dipakai (sebaiknya di-crop manual)')
        daftar[stem] = kartu
    return daftar


def bangun_db(method, template_gambar):
    db = OrderedDict()
    for nama, kartu in template_gambar.items():
        kps, des = fitur_kartu(kartu, method)
        if des is None or len(kps) < 20:
            print(f'[peringatan] template {nama} ({method}): hanya {0 if des is None else len(kps)} keypoint')
        db[nama] = dict(img=kartu, kps=kps, des=des, warna=warna_dari_nama(nama))
    return db


def fitur_query(kand, method, rot):
    # Fitur kartu pada rotasi tertentu; di-cache supaya eksperimen tidak menghitung ulang.
    kunci = (method, rot)
    if kunci not in kand['cache']:
        t0 = time.perf_counter()
        w = kand['warp'] if rot == 0 else putar(kand['warp'], rot)
        kps, des = fitur_kartu(w, method)
        kand['cache'][kunci] = (w, kps, des, time.perf_counter() - t0)
    return kand['cache'][kunci]


def kenali_kartu(kand, method, db, rasio, skor_mode, filter_warna):
    warna = warna_kartu(kand['warp']) if filter_warna else None
    terbaik, semua, waktu = None, {}, 0.0
    for rot in ORIENTASI:
        w, kps_q, des_q, t_fitur = fitur_query(kand, method, rot)
        waktu += t_fitur
        if des_q is None or len(kps_q) < 2:
            continue
        t0 = time.perf_counter()
        for nama_t, t in db.items():
            if warna is not None and t['warna'] != warna:
                continue
            if t['des'] is None or len(t['kps']) < 2:
                continue
            good = matchKeyPointsKNN(des_q, t['des'], rasio, method)
            inlier, status = 0, None
            if skor_mode == 'inlier' or len(good) > 4:
                M = getHomography(kps_q, t['kps'], des_q, t['des'], good, reprojThresh=RANSAC_THR)
                if M is not None and M[2] is not None:
                    status = M[2]
                    inlier = int(status.sum())
            skor = inlier if skor_mode == 'inlier' else len(good)
            semua[nama_t] = max(skor, semua.get(nama_t, -1))
            if terbaik is None or skor > terbaik['skor']:
                terbaik = dict(nama=nama_t, skor=skor, good=len(good), inlier=inlier, rot=rot,
                               kps_q=kps_q, kps_t=t['kps'], good_m=good, status=status,
                               warp=w, tmpl_img=t['img'], n_kp=len(kps_q))
        waktu += time.perf_counter() - t0
    if terbaik is None:
        terbaik = dict(nama=None, skor=0, good=0, inlier=0, rot=0, kps_q=(), kps_t=(), good_m=[],
                       status=None, warp=kand['warp'], tmpl_img=None, n_kp=0)
    urut = sorted(semua.values(), reverse=True)
    terbaik['skor2'] = urut[1] if len(urut) > 1 else 0
    terbaik['waktu'] = waktu
    return terbaik


def kenali_gambar(img, method, kandidat=None, rasio=None, skor_mode=None, filter_warna=None, ambang=None, db=None):
    rasio = RASIO if rasio is None else rasio
    skor_mode = SKOR_MODE if skor_mode is None else skor_mode
    filter_warna = FILTER_WARNA if filter_warna is None else filter_warna
    ambang = AMBANG[skor_mode][method] if ambang is None else ambang
    kandidat = deteksi_kartu(img) if kandidat is None else kandidat
    hasil = []
    for k in kandidat:
        h = kenali_kartu(k, method, (DB if db is None else db)[method], rasio, skor_mode, filter_warna)
        h['quad'] = k['quad']
        h['cadangan'] = k['cadangan']
        h['diterima'] = h['skor'] >= ambang
        hasil.append(h)
    return hasil


# ===================== TAMPILAN HASIL =====================
def gambar_hasil(img, hasil):
    out = img.copy()
    tebal = max(2, img.shape[1] // 350)
    skala = max(0.6, img.shape[1] / 1000 * 0.9)
    for h in hasil:
        q = h['quad'].astype(np.int32)
        warna = (0, 180, 0) if h['diterima'] else (0, 0, 255)
        cv2.polylines(out, [q], True, warna, tebal)
        label = f"{nama_tampil(h['nama'])} ({h['good']})" if h['diterima'] else f"tidak dikenali ({h['good']})"
        x, y = int(q[:, 0].min()), int(q[:, 1].min())
        pos = (max(x, 5), max(y - 8, 25))
        cv2.putText(out, label, pos, cv2.FONT_HERSHEY_SIMPLEX, skala, (255, 255, 255), tebal + 3, cv2.LINE_AA)
        cv2.putText(out, label, pos, cv2.FONT_HERSHEY_SIMPLEX, skala, warna, tebal, cv2.LINE_AA)
    return out


def gambar_match(h, maks=50):
    # drawMatches seperti di modul Prak 5; hanya padanan terbaik (jarak terkecil) yang digambar
    if h['tmpl_img'] is None or not h['good_m']:
        return None
    m = sorted(h['good_m'], key=lambda x: x.distance)[:maks]
    return cv2.drawMatches(h['warp'], h['kps_q'], h['tmpl_img'], h['kps_t'], m, None,
                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)




# ===================== DATABASE TEMPLATE & PROSES SATU FOTO =====================
DB = {}


def siapkan_database(template_dir=None, daftar=DAFTAR_DETEKTOR):
    # Muat 12 template, lalu hitung keypoint/descriptor ORB dan SIFT. Dipanggil sekali saat aplikasi mulai.
    global DB
    template = muat_template(template_dir)
    DB = {m: bangun_db(m, template) for m in daftar}
    return DB


def proses_foto(bytes_gambar, method=None, db=None):
    # Dari byte file foto -> (gambar asli, gambar hasil, hasil per kartu, waktu detik).
    method = method or DETEKTOR_UTAMA
    data = np.frombuffer(bytes_gambar, np.uint8)
    asli = cv2.imdecode(data, cv2.IMREAD_COLOR)          # EXIF orientasi HP diterapkan otomatis
    if asli is None:
        raise ValueError('File bukan gambar yang valid (pakai JPG/PNG).')
    t0 = time.perf_counter()
    img = resize_maks(asli)
    kand = deteksi_kartu(img, asli=asli)
    hasil = kenali_gambar(img, method, kandidat=kand, db=db)
    waktu = time.perf_counter() - t0
    return img, gambar_hasil(img, hasil), hasil, waktu
