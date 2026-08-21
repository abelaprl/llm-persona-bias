# Laporan Incomplete Generation pada Model Reasoning

Tanggal pemeriksaan: **12 Agustus 2026**

## Tujuan

Laporan ini membandingkan kegagalan setelah rerun yang terjadi ketika model
tidak sempat menghasilkan jawaban akhir. Pemeriksaan dilakukan terhadap file
percobaan rerun dan kondisi JSON utama terkini. Tidak ada API call yang dilakukan
saat penyusunan laporan.

Kasus ini disebut **incomplete generation**, bukan jawaban salah, karena request
berhasil diproses tetapi model tidak menghasilkan final answer yang dapat
dievaluasi.

## Ringkasan

| Model | Scope | Masih tanpa final answer | `finish_reason=length` | `content_filter` | Lainnya |
|---|---|---:|---:|---:|---:|
| GPT-OSS-20B | Nonregional | 6 | 6 | 0 | 0 |
| GPT-OSS-20B | Regional Cultural | 782 | 769 | 10 | 3 |
| **GPT-OSS-20B** | **Total** | **788** | **775** | **10** | **3** |
| Qwen3 Next 80B | Nonregional | 108 | 94 | 12 | 2 |
| Qwen3 Next 80B | Regional Cultural | 45 | 28 | 6 | 11 |
| **Qwen3 Next 80B** | **Total** | **153** | **122** | **18** | **13** |
| Llama 3.3 70B | Semua scope | 0 | 0 | 0 | 0 |
| Ministral 8B | Semua scope | 0 | 0 | 0 | 0 |
| Llama 3.1 8B | Semua scope | 0 | 0 | 0 | 0 |
| **Keseluruhan** |  | **941** | **897** | **28** | **16** |

GPT-OSS-20B merupakan model dengan incomplete generation paling banyak. Dari
5.676 target pada manifest rerun GPT-OSS, 4.888 telah menghasilkan final answer
dan 788 masih belum menghasilkan final answer. Proporsi incomplete generation
GPT-OSS setelah rerun adalah sekitar **13,88%**.

Qwen3 Next 80B juga menunjukkan pola serupa, tetapi jumlah akhirnya lebih kecil:
153 target masih tidak mempunyai final answer.

## Arti `finish_reason=length`

`finish_reason=length` berarti model mencapai batas token keluaran ketika masih
menghasilkan reasoning. Proses berhenti sebelum model sempat menulis jawaban
akhir pada field `content`.

Karakteristik kasus ini:

- endpoint dapat mengembalikan HTTP 200;
- reasoning tersedia;
- final content kosong;
- output tidak dapat dinilai benar atau salah;
- menaikkan `max_tokens` dapat membantu, tetapi meningkatkan biaya dan tidak
  menjamin final answer akan muncul.

Pada pemeriksaan ini, batas token merupakan penyebab **897 dari 941** incomplete
generation yang masih tersisa, atau sekitar **95,32%**.

## Arti kategori lainnya

### `content_filter`

Model atau provider menghentikan generasi karena filter konten. Kasus ini tidak
boleh diklasifikasikan sebagai jawaban salah karena final answer tidak tersedia.

### Lainnya

Kategori lainnya mencakup respons HTTP 200 yang tidak menyediakan final content
dalam format yang dapat digunakan serta kegagalan respons lain yang bukan
`length` atau `content_filter`.

## Perbandingan antarmodel

| Model | Kondisi setelah rerun |
|---|---|
| GPT-OSS-20B | 788 incomplete generation; masalah terbesar adalah reasoning mencapai batas token |
| Qwen3 Next 80B | 153 incomplete generation; pola sama tetapi lebih sedikit daripada GPT-OSS |
| Llama 3.3 70B | Semua kegagalan manifest telah terselesaikan setelah resume |
| Ministral 8B | Semua kegagalan manifest telah terselesaikan setelah resume |
| Llama 3.1 8B | Semua kegagalan manifest telah terselesaikan |

Temuan ini menunjukkan bahwa masalah final answer kosong tidak unik pada
GPT-OSS. Qwen3 Next 80B juga mengalaminya. Namun, frekuensinya jauh lebih tinggi
pada GPT-OSS-20B dalam konfigurasi rerun yang digunakan.

## Perlakuan dalam analisis TA

Sebanyak 941 output tersebut sebaiknya diberi label **incomplete generation**
atau **invalid response**, bukan dimasukkan sebagai jawaban salah.

Rekomendasi pelaporan:

1. Laporkan jumlah incomplete generation per model dan per skenario.
2. Keluarkan kasus tersebut dari denominator akurasi utama jika metrik yang
   digunakan adalah akurasi bersyarat pada respons valid.
3. Laporkan juga completion/valid-response rate agar pengeluaran data tidak
   menyembunyikan perbedaan reliabilitas antarmodel.
4. Jika seluruh item tetap digunakan sebagai denominator, tampilkan metrik
   tambahan yang secara eksplisit menghitung incomplete generation sebagai
   kegagalan sistem, bukan sebagai prediksi label yang salah.
5. Jangan mengambil label jawaban langsung dari reasoning tanpa aturan parsing
   dan validasi yang telah ditetapkan sebelumnya.

Metrik yang disarankan:

```text
valid_response_rate = jumlah output dengan final answer / jumlah target

conditional_accuracy = jumlah jawaban benar / jumlah output dengan jawaban valid

end_to_end_accuracy = jumlah jawaban benar / jumlah seluruh target
```

Dengan tiga angka tersebut, performa penalaran dan reliabilitas generasi dapat
dibedakan secara transparan.

## Keputusan apabila tidak dilakukan rerun tambahan

Tidak melakukan rerun tambahan dapat diterima apabila biaya menjadi kendala,
dengan syarat:

- 788 kasus GPT-OSS dan 153 kasus Qwen tetap dicatat;
- kasus tersebut tidak diubah menjadi jawaban benar atau salah;
- alasan exclusion dilaporkan;
- jumlah respons valid dan denominator setiap metrik dijelaskan;
- perbandingan model mempertimbangkan valid-response rate.

Jika diputuskan melakukan percobaan tambahan, target dapat direrun dengan batas
token lebih besar, misalnya 8.192. Percobaan tersebut harus dicatat sebagai
konfigurasi berbeda karena perubahan token budget dapat memengaruhi biaya dan
distribusi hasil.

## Sumber bukti lokal

Pemeriksaan menggunakan kondisi JSON utama terkini dan direktori berikut:

- `result/_rerun_attempts/gpt_oss_20b_nonregional`;
- `result/_rerun_attempts/gpt_oss_20b_regional`;
- `result/_rerun_attempts/qwen3_next_80b_nonregional`;
- `result/_rerun_attempts/qwen3_next_80b_regional`;
- `result/_rerun_attempts/llama33_70b_nonregional`;
- `result/_rerun_attempts/llama33_70b_regional`;
- `result/_rerun_attempts/ministral_8b_regional`.

Attempt lama yang kemudian mempunyai final answer sukses pada JSON utama tidak
dihitung sebagai kegagalan tersisa.
