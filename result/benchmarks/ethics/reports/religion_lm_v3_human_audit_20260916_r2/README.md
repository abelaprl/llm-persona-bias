# Paket recall manusia efektif

Menggantikan paket 20260915: seed kelompok yang sama sebelumnya menghasilkan 1.020 respons dari hanya 10 pertanyaan dan nol justice. Paket ini memakai stream RNG berbeda per model/persona/subset; total tetap 10 respons per model/persona.

Kuota per cell: commonsense=3, deontology=3, justice=2, virtue=2. Gunakan inverse inclusion probability pada sampling_design_key.csv. Split pertanyaan tidak diubah.

Dua reviewer mengisi file secara independen sebelum adjudikasi. Blank bukan label nol. Baca seluruh final, termasuk no-hit/abstain. visible_to_raw_offsets memetakan karakter tampilan ke offset final asli untuk anotasi span; raw_final_sha256 mengikat sumber. Jangan menyebut paket ini gold sebelum anotasi lengkap.
