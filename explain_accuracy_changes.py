import os
from pathlib import Path

print("\n" + "="*140)
print("PENJELASAN: MENGAPA PERSENTASE AKURASI BERUBAH-UBAH")
print("="*140 + "\n")

print("PENYEBAB #1: JUMLAH FILE YANG TERUS BERTAMBAH")
print("-" * 140)
print("""
Saat aku menjalankan analisis pertama kali:
- Grok man_implicits: 473 file → Akurasi: 93.93%
- Grok woman_implicits: 450 file → Akurasi: 93.59%

Sekarang (setelah proses berjalan lebih lama):
- Grok man_implicits: 508 file → Akurasi akan berbeda
- Grok woman_implicits: 481 file → Akurasi akan berbeda

Karena file terus bertambah (experiment masih running), akurasi bisa naik/turun
tergantung kualitas jawaban model di batch berikutnya.
""")

print("\nPENYEBAB #2: NEMOTRON INDEX MULAI DARI 1, BUKAN 0")
print("-" * 140)
print("""
Nemotron menyimpan file dari index 1, bukan 0:
- gsm8k_00001.json = Question index 0
- gsm8k_00002.json = Question index 1
- dst...

Tapi:
- Grok menyimpan dari index 300 (Q300-Q1087 range)
- Nebulon menyimpan dari index 300 (Q300-Q721 range)

Ini menyebabkan offset/misalignment jika tidak di-handle dengan benar.
""")

print("\nPENYEBAB #3: BERBAGAI VERSI SCRIPT ANALISIS")
print("-" * 140)
print("""
Aku membuat beberapa versi script yang berbeda:
1. analyze_results.py (early version) - parsing dari filename
2. detailed_accuracy_analysis.py - menggunakan question_index
3. final_comprehensive_analysis.py - mix up dengan folder structure
4. comprehensive_final_analysis.py - duplicate data
5. final_accurate_analysis.py - yang paling benar (sekarang)

Setiap versi bisa menghasilkan hasil berbeda karena logic berbeda.
""")

print("\nSOLUSI: CSV FILES SUDAH TERSIMPAN")
print("-" * 140)
print("""
Untung CSV sudah di-generate dan tersimpan! Itu adalah source of truth terbaik.
Karena CSV sudah di-save, angka di dalamnya adalah akurasi yang PALING AKURAT.

Lokasi CSV:
""")

csv_files = list(Path("results/reports").glob("*.csv"))
for i, csv_file in enumerate(sorted(csv_files), 1):
    size = csv_file.stat().st_size / 1024  # KB
    print(f"  {i}. {csv_file.name} ({size:.1f} KB)")

print("\n" + "="*140)
print("CARA LIHAT CSV")
print("="*140 + "\n")
print("""
OPSI 1: Buka langsung di Excel/LibreOffice
   Results → Reports → Pilih file .xlsx atau .csv

OPSI 2: Cat di terminal
   cd results/reports/
   head -50 "x-ai Grok 4.1 Fast__man_implicits.csv"

OPSI 3: Hitung accuracy dari CSV
   - Lihat kolom "Correct"
   - Hitung: (jumlah "Yes") / (total rows) × 100

OPSI 4: Gunakan pandas
   import pandas as pd
   df = pd.read_csv("results/reports/x-ai Grok 4.1 Fast__man_implicits.csv")
   accuracy = (df['Correct'] == 'Yes').sum() / len(df) * 100
""")

print("\n" + "="*140)
print("KESIMPULAN")
print("="*140 + "\n")
print("""
Persentase berubah-ubah karena:
1. File hasil eksperimen terus bertambah (proses masih running)
2. Ada multiple script versi dengan logic berbeda
3. Indexing/offset yang berbeda antara model (Nemotron vs Grok vs Nebulon)

✓ CSV FILES adalah TRUTH SOURCE karena sudah di-save
✓ Angka di CSV TIDAK AKAN BERUBAH (sudah final)
✓ Akurasi baru hanya muncul jika kita re-run analysis dengan FILE LEBIH BANYAK
""")

