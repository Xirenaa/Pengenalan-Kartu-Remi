"""Menjalankan pipeline pada 28 foto uji dan mencetak akurasi (sama seperti bagian 5 notebook).

Pemakaian:  python evaluasi.py
"""
import csv
import os
from collections import defaultdict

import kartu_remi as kr


def baca_gt(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r['set_kartu'] = {x.strip() for x in r['kartu'].split('|') if x.strip()}
    return rows


def main():
    rows = baca_gt(os.path.join(kr.DATASET_DIR, 'ground_truth.csv'))
    kr.siapkan_database()
    for method in kr.DAFTAR_DETEKTOR:
        grup = defaultdict(lambda: [0, 0])
        for r in rows:
            with open(os.path.join(kr.DATASET_DIR, r['file']), 'rb') as f:
                _, _, hasil, _ = kr.proses_foto(f.read(), method)
            pred = {h['nama'] for h in hasil if h['diterima']}
            ok = pred == r['set_kartu']
            for k in (r['kondisi'], 'SEMUA'):
                grup[k][0] += int(ok)
                grup[k][1] += 1
        print(f'--- {method.upper()} ---')
        for k in ('rotasi', 'skala', 'cahaya', 'sulit', 'SEMUA'):
            b, t = grup[k]
            print(f'{k:<8}: {b}/{t} ({100 * b / t:.1f}%)')


if __name__ == '__main__':
    main()
