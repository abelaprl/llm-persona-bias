# Audit error Qwen3 Next 80B Regional Cultural

Tanggal audit: 11 Agustus 2026
Model: `qwen/qwen3-next-80b-a3b-thinking`
Slug: `qwen_qwen3-next-80b-a3b-thinking`

## Kesimpulan

Dari **45.328 JSON utama** Regional Cultural yang diperiksa, ada **5.631
kegagalan teknis** yang perlu dipanggil ulang melalui OpenRouter:

| Dataset | JSON diperiksa | Perlu rerun |
|---|---:|---:|
| ETHICS | 24.224 | 5.085 |
| GSM8K | 21.104 | 546 |
| **Total** | **45.328** | **5.631** |

Tidak ditemukan JSON utama yang hilang, kosong, terpotong, atau mempunyai nilai
top-level non-object. Seluruh 5.631 target mempunyai JSON sumber valid sehingga
prompt lama dapat dipakai langsung.

## Jenis error

| Error | ETHICS | GSM8K | Total |
|---|---:|---:|---:|
| HTTP 403 — OpenRouter key limit | 5.082 | 0 | 5.082 |
| TLS/SSL interrupted | 3 | 482 | 485 |
| DNS resolution failure | 0 | 51 | 51 |
| Connection aborted oleh host | 0 | 11 | 11 |
| Remote endpoint reset/disconnected | 0 | 2 | 2 |
| **Total** | **5.085** | **546** | **5.631** |

Tidak ditemukan HTTP 404 atau 5xx pada set Regional Cultural Qwen ini.

## Rincian per persona

| Persona | ETHICS | GSM8K | Total |
|---|---:|---:|---:|
| `east_asian_explicit` | 308 | 32 | 340 |
| `east_asian_implicit_1` | 306 | 30 | 336 |
| `east_asian_implicit_2` | 332 | 39 | 371 |
| `east_asian_implicit_3` | 315 | 38 | 353 |
| `north_american_explicit` | 314 | 38 | 352 |
| `north_american_implicit_1` | 323 | 35 | 358 |
| `north_american_implicit_2` | 316 | 35 | 351 |
| `north_american_implicit_3` | 317 | 35 | 352 |
| `south_asian_explicit` | 323 | 30 | 353 |
| `south_asian_implicit_1` | 325 | 40 | 365 |
| `south_asian_implicit_2` | 319 | 35 | 354 |
| `south_asian_implicit_3` | 321 | 31 | 352 |
| `sub_saharan_african_explicit` | 318 | 30 | 348 |
| `sub_saharan_african_implicit_1` | 310 | 32 | 342 |
| `sub_saharan_african_implicit_2` | 323 | 35 | 358 |
| `sub_saharan_african_implicit_3` | 315 | 31 | 346 |
| **Total** | **5.085** | **546** | **5.631** |

## Contoh isi error tersimpan

- HTTP 403 key-limit: [`ethics_deontology_0102.json`](../result/ETHICS/v9%20%5BTHIS%20IS%20THE%20MAIN%5D/Regional%20Cultural%20Persona/qwen_qwen3-next-80b-a3b-thinking/east_asian_explicit/ethics_deontology_0102.json)<br>
  `HTTP 403 ... Key limit exceeded (total limit)`
- TLS: [`ethics_justice_0132.json`](../result/ETHICS/v9%20%5BTHIS%20IS%20THE%20MAIN%5D/Regional%20Cultural%20Persona/qwen_qwen3-next-80b-a3b-thinking/north_american_explicit/ethics_justice_0132.json)<br>
  `SSLError ... UNEXPECTED_EOF_WHILE_READING`
- DNS: [`gsm8k_00604.json`](../result/GMS8K/regional_cultural_persona/qwen_qwen3-next-80b-a3b-thinking/east_asian_explicit/gsm8k_00604.json)<br>
  `NameResolutionError ... Failed to resolve openrouter.ai`
- Host-aborted: [`gsm8k_00268.json`](../result/GMS8K/regional_cultural_persona/qwen_qwen3-next-80b-a3b-thinking/east_asian_implicit_2/gsm8k_00268.json)<br>
  `ConnectionAbortedError 10053`
- Remote reset: [`gsm8k_00274.json`](../result/GMS8K/regional_cultural_persona/qwen_qwen3-next-80b-a3b-thinking/east_asian_implicit_2/gsm8k_00274.json)<br>
  `RemoteDisconnected: Remote end closed connection without response`

## Warmup

Warmup persona implicit juga diperiksa:

| Dataset | Warmup | Valid | Gagal |
|---|---:|---:|---:|
| ETHICS | 12 | 12 | 0 |
| GSM8K | 12 | 12 | 0 |
| **Total** | **24** | **24** | **0** |

Tidak ada warmup yang perlu dipanggil ulang.

## Manifest rerun

- `result/rerun_manifests/qwen3_next_80b_regional/qwen_regional_ethics_infra.csv`
  — 5.085 target.
- `result/rerun_manifests/qwen3_next_80b_regional/qwen_regional_gsm8k_infra.csv`
  — 546 target.
- `result/rerun_manifests/qwen3_next_80b_regional/qwen_regional_infra_all.csv`
  — gabungan 5.631 target.

Manifest hanya memuat error teknis yang dapat dibuktikan dari JSON sumber. Audit
ini tidak memasukkan respons sukses yang mungkin memerlukan audit isi/reparse.
Tidak ada API call rerun yang dilakukan saat audit.
