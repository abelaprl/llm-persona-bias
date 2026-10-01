"""Penilaian — DIKUNCI, identik untuk semua model dan semua grup persona.

Bab IV 1275-1276: respons yang sudah diperoleh tetapi tidak memuat jawaban akhir
setelah seluruh aturan diterapkan dinilai SALAH. Tidak ada baris yang dibuang
dari penyebut akurasi.
"""
from .numbers import answers_equal, parse_gold


def score(extract_result, ground_truth_text):
    gold = parse_gold(ground_truth_text)
    correct = answers_equal(extract_result.value, gold) if extract_result.value is not None else False
    return gold, correct
