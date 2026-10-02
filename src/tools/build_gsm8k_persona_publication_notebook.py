"""Build the six-main-model GSM8K Gender/Religion/Regional publication notebook."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = ROOT / "result" / "GMS8K" / "reports" / "GSM8K_Gender_Religion_Publication_Analysis.ipynb"


def md(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}

nb["cells"] = [
    md(
        r"""
# Analisis Persona Gender, Religion, dan Regional Cultural pada GSM8K — Enam Model Utama

**Versi publikasi/skripsi: akurasi, effect size, significance value, dan token**

Notebook ini hanya menggunakan enam model utama:

1. Qwen3 Next 80B-A3B Thinking
2. Llama 3.1 8B Instruct
3. Ministral 8B
4. DeepSeek R1-Distill Qwen 32B
5. GPT-OSS 20B
6. Llama 3.3 70B Instruct

Sumber Gender dan Religion berasal dari dua CSV agregat berikut:

- `result/GMS8K/reports/akurasi gender&neutral persona.csv`
- `result/GMS8K/reports/akurasi religion persona.csv`

Regional Cultural menggunakan enam `results.csv` tingkat pertanyaan dari report canonical
Regional Cultural yang sudah tervalidasi di `reports/`.

Daftar putih model dan validasi otomatis mencegah model di luar keenam model tersebut masuk ke tabel atau gambar.

## Posisi metodologis

Kedua CSV Gender/Religion berisi agregat per model-persona, bukan hasil benar/salah per
`question_id`. Oleh karena itu untuk dua dataset tersebut:

- effect size yang dapat dihitung secara sah adalah **risk difference**, **Cohen's h**, dan **marginal odds ratio**;
- token dianalisis sebagai rerata, selisih, dan rasio deskriptif;
- McNemar, paired bootstrap, dan Wilcoxon **tidak dihitung**, karena ketiganya memerlukan data tingkat pertanyaan;
- p-value independen juga tidak dipakai sebagai bukti utama karena seluruh persona mengerjakan pertanyaan GSM8K yang sama.

Regional Cultural berbeda: hasil per-`question_id` tersedia lengkap. Perbandingan granular
Implicit versus Explicit memakai **exact McNemar**, lalu seluruh p-value Regional dikoreksi
dengan **Benjamini–Hochberg FDR**. Effect size tetap dilaporkan bersama significance value.

Keputusan ini lebih defensibel untuk TA daripada memperlakukan hasil berpasangan sebagai sampel independen.

## Dasar desain visual

Paket visual mengikuti dua sumber: struktur delta/effect-size pada dokumen referensi lokal dan praktik paper persona-demographic evaluation:

- [Lutz et al. (EMNLP Findings 2025)](https://aclanthology.org/2025.findings-emnlp.1261/) memisahkan perbandingan kelompok dan strategi prompt, memakai panel kecil serta agregat lintas-model untuk menjaga hierarki bacaan.
- [Sorokovikova et al. (GeBNLP 2025)](https://aclanthology.org/2025.gebnlp-1.20/) menampilkan fraksi performa per persona berdampingan dengan kondisi dasar dan menggunakan McNemar hanya ketika data pasangan tersedia.
- [Sun et al. (NAACL 2025)](https://aclanthology.org/2025.naacl-short.71/) mengutamakan perubahan terhadap reference condition pada sumbu nol, sehingga arah dan besar perubahan langsung terbaca.
- [Hu dan Collier (ACL 2024)](https://aclanthology.org/2024.acl-long.554/) menggunakan heatmap ketika kedua sumbu benar-benar kategorikal dan scatter hanya untuk relasi kuantitatif yang memang diuji.

Implikasinya untuk notebook ini: level absolut ditampilkan lebih dahulu, lalu delta terhadap baseline/framing, kemudian effect size; heatmap dipakai untuk matriks perbandingan, bukan sebagai pengganti semua jenis grafik.
"""
    ),
    code(
        r"""
from __future__ import annotations

import csv
import hashlib
import itertools
import math
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Image, Markdown, display
from scipy.stats import binomtest

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_columns", 80)
pd.set_option("display.max_rows", 150)
pd.set_option("display.float_format", lambda value: f"{value:,.4f}")

EXPECTED_GSM8K_N = 1_319

def find_repo_root(start: Path | None = None) -> Path:
    candidate = Path(start or Path.cwd()).resolve()
    for path in [candidate, *candidate.parents]:
        if (path / "result" / "GMS8K").exists() and (path / "docs").exists():
            return path
    raise FileNotFoundError("Repo root tidak ditemukan. Jalankan dari D:/TugasAkhir atau subfoldernya.")

REPO_ROOT = find_repo_root()
REPORT_ROOT = REPO_ROOT / "result" / "GMS8K" / "reports"
GENDER_CSV = REPORT_ROOT / "akurasi gender&neutral persona.csv"
RELIGION_CSV = REPORT_ROOT / "akurasi religion persona.csv"
REGIONAL_REPORT_ROOT = REPORT_ROOT / "Regional_Cultural_Persona"
OUTPUT_ROOT = REPORT_ROOT / "persona_publication_analysis_v4_with_regional_six_models"
TABLE_DIR = OUTPUT_ROOT / "tables"
FIGURE_DIR = OUTPUT_ROOT / "figures"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# This directory is exclusively generated by this notebook. Clear previous
# figure renders so removed/renumbered figures cannot survive as stale files.
for previous_figure in [*FIGURE_DIR.glob("*.png"), *FIGURE_DIR.glob("*.svg")]:
    previous_figure.unlink()
for obsolete_table in [
    TABLE_DIR / "coverage_summary_six_models.csv",
    TABLE_DIR / "regional_commit_rate_by_persona.csv",
]:
    if obsolete_table.exists():
        obsolete_table.unlink()

MODEL_ORDER = [
    "qwen_qwen3-next-80b-a3b-thinking",
    "meta-llama_llama-3.1-8b-instruct",
    "mistralai_ministral-8b-2512",
    "deepseek_deepseek-r1-distill-qwen-32b",
    "openai_gpt-oss-20b",
    "meta-llama_llama-3.3-70b-instruct",
]
MODEL_LABELS = {
    "qwen_qwen3-next-80b-a3b-thinking": "Qwen3 Next 80B-A3B",
    "meta-llama_llama-3.1-8b-instruct": "Llama 3.1 8B",
    "mistralai_ministral-8b-2512": "Ministral 8B",
    "deepseek_deepseek-r1-distill-qwen-32b": "DeepSeek R1-Distill 32B",
    "openai_gpt-oss-20b": "GPT-OSS 20B",
    "meta-llama_llama-3.3-70b-instruct": "Llama 3.3 70B",
}
GENDER_MODEL_ALIASES = {
    "Qwen3 Next 80B A3B Thinking": "qwen_qwen3-next-80b-a3b-thinking",
    "Meta Llama 3.1 8B Instruct": "meta-llama_llama-3.1-8b-instruct",
    "Mistral Ministral 8B": "mistralai_ministral-8b-2512",
    "deepseek-r1-distill-qwen-32b": "deepseek_deepseek-r1-distill-qwen-32b",
    "OpenAI GPT-OSS-20B": "openai_gpt-oss-20b",
    "Meta Llama 3.3 70B Instruct": "meta-llama_llama-3.3-70b-instruct",
}
PERSONA_PATTERN = r"^(man|woman|neutral|atheist|christian|islam|jewish|north_american|east_asian|south_asian|sub_saharan_african)(?:_(explicit|implicit)(?:_(\d+))?)?$"

REGIONAL_RESULTS = {
    model: REGIONAL_REPORT_ROOT / model / "results.csv"
    for model in MODEL_ORDER
}

sns.set_theme(context="paper", style="whitegrid", palette="colorblind", font_scale=1.05)
mpl.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})

# Publication color system: saturated, consistent across equivalent chart types.
ACCURACY_COLORS = [
    "#0057B8", "#0096C7",  # identity 1: explicit, implicit average
    "#D14900", "#F57C00",  # identity 2
    "#087F23", "#2EAD4A",  # identity 3
    "#6A1B9A", "#9C27B0",  # identity 4
    "#3F3F3F",              # neutral
]
DELTA_COLORS = ["#0067B9", "#E05A00", "#138A36", "#7B1FA2"]
GENDER_DELTA_COLORS = ["#0067B9", "#C62828", "#6A1B9A"]
EFFECT_CMAP = "RdBu_r"
TOKEN_CMAP = "viridis"
TOKEN_DELTA_CMAP = "PuOr"
SIGNIFICANCE_CMAP = "Reds_r"

print(f"Repo       : {REPO_ROOT}")
print(f"Gender CSV : {GENDER_CSV}")
print(f"Religion CSV: {RELIGION_CSV}")
print(f"Regional question-level sources: {len(REGIONAL_RESULTS)} files")
print(f"Output baru: {OUTPUT_ROOT}")
"""
    ),
    md(
        r"""
## 1. Loader sumber Gender, Religion, dan Regional Cultural

CSV Gender merupakan kumpulan blok dengan header yang berulang dan skema yang sedikit berbeda antar-model. Loader hanya menerima baris dengan persona valid serta nama model yang berada dalam daftar putih. CSV Religion sudah berbentuk tabel rapi dan difilter dengan daftar putih yang sama.

Regional Cultural dibaca dari enam `results.csv` tingkat pertanyaan. Loader memeriksa tepat
16 persona x 1.319 soal per model, keunikan `model/persona/question_id`, model sumber,
status jawaban, serta kolom token sebelum membuat agregat publikasi.
"""
    ),
    code(
        r"""
NUMERIC_COLUMNS = [
    "correct", "total", "accuracy", "committed", "commit_rate", "abstain", "missing",
    "candidate_only", "accuracy_on_committed", "sum_prompt_tokens", "sum_completion_tokens",
    "sum_reasoning_tokens", "sum_total_tokens", "avg_prompt_tokens", "avg_completion_tokens",
    "avg_reasoning_tokens", "avg_total_tokens",
]

def sha256_file(path: Path, chunk_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()

def parse_number(value: str | float | int | None) -> float:
    if value is None or str(value).strip() == "":
        return np.nan
    return float(str(value).strip())

def load_gender_aggregate(path: Path) -> pd.DataFrame:
    records = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            padded = row + [""] * max(0, 16 - len(row))
            raw_model, persona = padded[0].strip(), padded[1].strip()
            if raw_model not in GENDER_MODEL_ALIASES:
                continue
            if not pd.Series([persona]).str.match(PERSONA_PATTERN).iloc[0]:
                continue
            records.append({
                "dataset": "Gender",
                "model": GENDER_MODEL_ALIASES[raw_model],
                "source_model_name": raw_model,
                "persona": persona,
                "correct": parse_number(padded[2]),
                "total": parse_number(padded[3]),
                "accuracy": parse_number(padded[4]),
                "sum_prompt_tokens": parse_number(padded[8]),
                "sum_completion_tokens": parse_number(padded[9]),
                "sum_reasoning_tokens": parse_number(padded[10]),
                "sum_total_tokens": parse_number(padded[11]),
                "avg_prompt_tokens": parse_number(padded[12]),
                "avg_completion_tokens": parse_number(padded[13]),
                "avg_reasoning_tokens": parse_number(padded[14]),
                "avg_total_tokens": parse_number(padded[15]),
            })
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("Tidak ada baris enam model utama yang terbaca dari CSV Gender.")
    # Gender stores accuracy as a fraction. Preserve it as the requested primary
    # source column; count-derived accuracy is retained separately for audit.
    frame["accuracy_source_pct"] = 100 * frame["accuracy"]
    frame["accuracy_from_counts_pct"] = 100 * frame["correct"] / frame["total"]
    frame["accuracy_pct"] = frame["accuracy_source_pct"]
    frame["evaluated"] = frame["total"]
    frame["evaluated_rate_pct"] = 100 * frame["total"] / EXPECTED_GSM8K_N
    # The aggregate Gender file does not expose committed/abstain/missing separately.
    frame["committed"] = np.nan
    frame["commit_rate_pct"] = np.nan
    frame["abstain"] = np.nan
    frame["missing"] = EXPECTED_GSM8K_N - frame["total"]
    frame["candidate_only"] = np.nan
    frame["accuracy_committed_pct"] = np.nan
    return frame

def load_religion_aggregate(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[
        frame["model"].isin(MODEL_ORDER)
        & ~frame["persona"].astype(str).str.upper().isin({"OVERALL", "ALL"})
    ].copy()
    for column in NUMERIC_COLUMNS:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["dataset"] = "Religion"
    frame["source_model_name"] = frame["model"]
    frame["accuracy_source_pct"] = frame["accuracy"]
    frame["accuracy_from_counts_pct"] = 100 * frame["correct"] / frame["total"]
    frame["accuracy_pct"] = frame["accuracy_source_pct"]
    frame["evaluated"] = frame["total"]
    frame["evaluated_rate_pct"] = 100 * frame["total"] / EXPECTED_GSM8K_N
    frame["commit_rate_pct"] = frame["commit_rate"]
    frame["accuracy_committed_pct"] = frame["accuracy_on_committed"]
    return frame

REGIONAL_IDENTITIES = [
    "north_american", "east_asian", "south_asian", "sub_saharan_african",
]
REGIONAL_PERSONAS = [
    f"{identity}_{framing}{suffix}"
    for identity in REGIONAL_IDENTITIES
    for framing, suffix in [
        ("explicit", ""), ("implicit", "_1"), ("implicit", "_2"), ("implicit", "_3"),
    ]
]
REGIONAL_USECOLS = [
    "model", "persona", "question_id", "is_correct", "answer_status", "technical_failure",
    "prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens",
]

def normalize_bool(series: pd.Series, column: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    normalized = series.astype("string").str.strip().str.lower()
    unexpected = set(normalized.dropna().unique()) - {"true", "false", "1", "0"}
    if unexpected:
        raise ValueError(f"{column}: nilai boolean tidak dikenal: {sorted(unexpected)}")
    return normalized.map({"true": True, "1": True, "false": False, "0": False}).fillna(False).astype(bool)

def load_regional_question_level(paths: dict[str, Path]) -> pd.DataFrame:
    frames = []
    for expected_model in MODEL_ORDER:
        path = paths[expected_model]
        if not path.exists():
            raise FileNotFoundError(f"Regional source tidak ditemukan: {path}")
        frame = pd.read_csv(path, usecols=REGIONAL_USECOLS, dtype={"question_id": "string"})
        source_models = set(frame["model"].dropna().astype(str))
        assert source_models == {expected_model}, f"Model Regional salah pada {path}: {source_models}"
        assert len(frame) == 16 * EXPECTED_GSM8K_N, f"{expected_model}: Regional rows={len(frame)}"
        assert set(frame["persona"].astype(str)) == set(REGIONAL_PERSONAS)
        assert frame.groupby("persona").size().eq(EXPECTED_GSM8K_N).all()
        assert not frame.duplicated(["model", "persona", "question_id"]).any()
        assert frame["question_id"].notna().all()
        frame["is_correct"] = normalize_bool(frame["is_correct"], "is_correct")
        frame["technical_failure"] = normalize_bool(frame["technical_failure"], "technical_failure")
        for metric in ["prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens"]:
            frame[metric] = pd.to_numeric(frame[metric], errors="raise")
        frame["model"] = expected_model
        frame["source_path"] = str(path)
        frames.append(frame)
    detail = pd.concat(frames, ignore_index=True)
    assert len(detail) == 6 * 16 * EXPECTED_GSM8K_N
    assert not detail.duplicated(["model", "persona", "question_id"]).any()
    return detail

def aggregate_regional(detail: pd.DataFrame) -> pd.DataFrame:
    frame = (
        detail.assign(
            committed_flag=detail["answer_status"].eq("committed"),
            abstain_flag=detail["answer_status"].eq("abstain"),
            missing_flag=detail["answer_status"].eq("missing_content"),
            candidate_only_flag=detail["answer_status"].eq("candidate_only"),
        )
        .groupby(["model", "persona"], as_index=False)
        .agg(
            correct=("is_correct", "sum"), total=("question_id", "size"),
            committed=("committed_flag", "sum"), abstain=("abstain_flag", "sum"),
            missing=("missing_flag", "sum"), candidate_only=("candidate_only_flag", "sum"),
            technical_failures=("technical_failure", "sum"),
            sum_prompt_tokens=("prompt_tokens", "sum"),
            sum_completion_tokens=("completion_tokens", "sum"),
            sum_reasoning_tokens=("reasoning_tokens", "sum"),
            sum_total_tokens=("total_tokens", "sum"),
        )
    )
    frame["dataset"] = "Regional Cultural"
    frame["source_model_name"] = frame["model"]
    frame["accuracy"] = 100 * frame["correct"] / frame["total"]
    frame["accuracy_source_pct"] = frame["accuracy"]
    frame["accuracy_from_counts_pct"] = frame["accuracy"]
    frame["accuracy_pct"] = frame["accuracy"]
    frame["evaluated"] = frame["total"]
    frame["evaluated_rate_pct"] = 100 * frame["total"] / EXPECTED_GSM8K_N
    frame["commit_rate"] = 100 * frame["committed"] / frame["total"]
    frame["commit_rate_pct"] = frame["commit_rate"]
    frame["accuracy_on_committed"] = 100 * frame["correct"] / frame["committed"]
    frame["accuracy_committed_pct"] = frame["accuracy_on_committed"]
    for component in ["prompt", "completion", "reasoning", "total"]:
        frame[f"avg_{component}_tokens"] = frame[f"sum_{component}_tokens"] / frame["total"]
    return frame

gender = load_gender_aggregate(GENDER_CSV)
religion = load_religion_aggregate(RELIGION_CSV)
regional_detail = load_regional_question_level(REGIONAL_RESULTS)
regional = aggregate_regional(regional_detail)
summary = pd.concat([gender, religion, regional], ignore_index=True, sort=False)
summary["model_label"] = summary["model"].map(MODEL_LABELS)

parsed = summary["persona"].str.extract(PERSONA_PATTERN)
summary["identity"] = parsed[0]
summary["framing"] = parsed[1].fillna("neutral")
summary["variant"] = pd.to_numeric(parsed[2], errors="coerce").fillna(0).astype(int)

model_dtype = pd.CategoricalDtype(MODEL_ORDER, ordered=True)
summary["model"] = summary["model"].astype(model_dtype)
summary = summary.sort_values(["dataset", "model", "persona"]).reset_index(drop=True)

expected_models = set(MODEL_ORDER)
for dataset, subset in summary.groupby("dataset"):
    actual_models = set(subset["model"].astype(str))
    assert actual_models == expected_models, f"{dataset}: model tidak tepat enam: {actual_models}"
assert len(gender) == 6 * 9, f"Gender seharusnya 54 baris, ditemukan {len(gender)}"
assert len(religion) == 6 * 16, f"Religion seharusnya 96 baris, ditemukan {len(religion)}"
assert len(regional) == 6 * 16, f"Regional seharusnya 96 baris, ditemukan {len(regional)}"
assert len(summary) == 246
assert summary["model_label"].notna().all()
assert not summary.duplicated(["dataset", "model", "persona"]).any()

manifest_rows = [
    {"dataset": "Gender", "path": GENDER_CSV},
    {"dataset": "Religion", "path": RELIGION_CSV},
] + [
    {"dataset": "Regional Cultural", "model": model, "path": REGIONAL_RESULTS[model]}
    for model in MODEL_ORDER
]
manifest = pd.DataFrame(manifest_rows)
manifest["relative_path"] = manifest["path"].map(lambda p: str(p.relative_to(REPO_ROOT)).replace("\\", "/"))
manifest["bytes"] = manifest["path"].map(lambda p: p.stat().st_size)
manifest["mtime_iso"] = manifest["path"].map(lambda p: pd.Timestamp(p.stat().st_mtime, unit="s").isoformat())
manifest["sha256"] = manifest["path"].map(sha256_file)
manifest.drop(columns="path").to_csv(TABLE_DIR / "source_manifest.csv", index=False)

print(f"Gender rows : {len(gender)} = 6 model × 9 persona")
print(f"Religion rows: {len(religion)} = 6 model × 16 persona")
print(f"Regional rows: {len(regional)} agregat; {len(regional_detail):,} question-level")
print("Model yang dipakai:")
for model in MODEL_ORDER:
    print(f"- {MODEL_LABELS[model]}")
display(manifest.drop(columns="path"))
"""
    ),
    code(
        r"""
def wilson_interval(proportion: float, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    denominator = 1 + z**2 / n
    center = (proportion + z**2 / (2 * n)) / denominator
    half = z * math.sqrt(proportion * (1 - proportion) / n + z**2 / (4 * n**2)) / denominator
    return 100 * (center - half), 100 * (center + half)

intervals = summary.apply(
    lambda row: wilson_interval(float(row["accuracy_pct"]) / 100, int(row["total"])),
    axis=1, result_type="expand"
)
summary[["accuracy_ci95_low_pct", "accuracy_ci95_high_pct"]] = intervals
summary["accuracy_count_diff_pp"] = summary["accuracy_from_counts_pct"] - summary["accuracy_source_pct"]

summary_columns = [
    "dataset", "model", "model_label", "persona", "identity", "framing", "variant",
    "correct", "total", "accuracy_pct", "accuracy_ci95_low_pct", "accuracy_ci95_high_pct",
    "evaluated", "evaluated_rate_pct", "committed", "commit_rate_pct", "abstain", "missing",
    "candidate_only", "accuracy_committed_pct", "sum_prompt_tokens", "sum_completion_tokens",
    "sum_reasoning_tokens", "sum_total_tokens", "avg_prompt_tokens", "avg_completion_tokens",
    "avg_reasoning_tokens", "avg_total_tokens", "accuracy_source_pct", "accuracy_from_counts_pct",
    "accuracy_count_diff_pp",
]
summary[summary_columns].to_csv(TABLE_DIR / "accuracy_by_persona_six_models.csv", index=False)

inventory = (
    summary.groupby(["dataset", "model", "model_label"], observed=True, as_index=False)
    .agg(
        personas=("persona", "nunique"),
        rows=("persona", "size"),
        total_evaluated=("total", "sum"),
        total_correct=("correct", "sum"),
        min_evaluated_rate_pct=("evaluated_rate_pct", "min"),
        max_evaluated_rate_pct=("evaluated_rate_pct", "max"),
        token_personas_available=("avg_total_tokens", "count"),
    )
)
weighted_accuracy = (
    summary.assign(weighted_accuracy=lambda frame: frame["accuracy_pct"] * frame["total"])
    .groupby(["dataset", "model"], observed=True)["weighted_accuracy"].sum()
    / summary.groupby(["dataset", "model"], observed=True)["total"].sum()
).rename("weighted_accuracy_source_pct").reset_index()
inventory = inventory.merge(weighted_accuracy, on=["dataset", "model"], validate="one_to_one")
inventory["accuracy_from_reported_counts_pct"] = 100 * inventory["total_correct"] / inventory["total_evaluated"]
inventory.to_csv(TABLE_DIR / "data_inventory_six_models.csv", index=False)

source_audit = summary[[
    "dataset", "model", "persona", "correct", "total", "accuracy_pct", "accuracy_source_pct",
    "accuracy_from_counts_pct", "accuracy_count_diff_pp", "evaluated_rate_pct", "avg_total_tokens",
]].copy()
source_audit["accuracy_count_consistent"] = source_audit["accuracy_count_diff_pp"].abs().lt(0.051)
source_audit["token_available"] = source_audit["avg_total_tokens"].notna()
source_audit.to_csv(TABLE_DIR / "source_consistency_audit.csv", index=False)

# Known source anomaly: all nine Qwen Gender rows have stale/inconsistent
# correct counts relative to the accuracy column. Keep the requested accuracy
# values and expose the discrepancy; fail if any other block is inconsistent.
inconsistent = source_audit[~source_audit["accuracy_count_consistent"]]
assert len(inconsistent) == 9
assert set(inconsistent["dataset"]) == {"Gender"}
assert set(inconsistent["model"].astype(str)) == {"qwen_qwen3-next-80b-a3b-thinking"}
assert summary.groupby("dataset", observed=True)["model"].nunique().eq(6).all()

display(inventory.style.format(precision=3))
print(f"Anomali correct/total vs accuracy: {len(inconsistent)} baris (seluruhnya Qwen Gender)")
print(f"Max |selisih|: {source_audit['accuracy_count_diff_pp'].abs().max():.4f} pp")
"""
    ),
    md(
        r"""
## 2. Akurasi per persona

`accuracy_pct` mempertahankan kolom akurasi dari CSV sumber. Audit menemukan bahwa sembilan baris Qwen Gender mempunyai `correct/total` yang tidak cocok dengan kolom tersebut; notebook tidak menyembunyikan masalah ini dan menyimpan selisihnya di `source_consistency_audit.csv`. Karena pengguna meminta kolom akurasi tidak diubah, visualisasi dan effect size memakai kolom akurasi sumber, sedangkan hitungan benar ditampilkan hanya sebagai provenance. Wilson interval 95% menggunakan akurasi sumber dan denominator `total` sebagai ringkasan aproksimasi.

- Pada Gender, CSV tidak mempunyai kolom `committed`, `abstain`, atau `candidate_only`; karena itu notebook hanya melaporkan `evaluated_rate_pct = total / 1.319` dan tidak mengarang commit rate.
- Pada Religion, `commit_rate` dan `accuracy_on_committed` dipertahankan dari CSV sumber.
"""
    ),
    code(
        r"""
gender_order = [
    "neutral", "man_explicit_1", "woman_explicit_1",
    "man_implicit_1", "woman_implicit_1", "man_implicit_2", "woman_implicit_2",
    "man_implicit_3", "woman_implicit_3",
]
religion_order = [
    f"{identity}_{framing}{suffix}"
    for identity in ["atheist", "christian", "islam", "jewish"]
    for framing, suffix in [("explicit", ""), ("implicit", "_1"), ("implicit", "_2"), ("implicit", "_3")]
]
regional_order = [
    f"{identity}_{framing}{suffix}"
    for identity in REGIONAL_IDENTITIES
    for framing, suffix in [("explicit", ""), ("implicit", "_1"), ("implicit", "_2"), ("implicit", "_3")]
]

for dataset, order in [
    ("Gender", gender_order), ("Religion", religion_order), ("Regional Cultural", regional_order),
]:
    display(Markdown(f"### {dataset}: accuracy (%)"))
    table = (
        summary[summary["dataset"].eq(dataset)]
        .pivot(index="model_label", columns="persona", values="accuracy_pct")
        .reindex(index=[MODEL_LABELS[model] for model in MODEL_ORDER], columns=order)
    )
    display(table.style.format("{:.2f}").background_gradient(cmap="YlGn", vmin=75, vmax=100))

grouped_descriptive = (
    summary.groupby(["dataset", "model", "model_label", "identity", "framing"], observed=True, as_index=False)
    .agg(
        persona_variants=("persona", "nunique"),
        accuracy_persona_mean_pct=("accuracy_pct", "mean"),
        accuracy_persona_min_pct=("accuracy_pct", "min"),
        accuracy_persona_max_pct=("accuracy_pct", "max"),
        evaluated_rate_persona_mean_pct=("evaluated_rate_pct", "mean"),
        commit_rate_persona_mean_pct=("commit_rate_pct", "mean"),
        avg_prompt_tokens_persona_mean=("avg_prompt_tokens", "mean"),
        avg_completion_tokens_persona_mean=("avg_completion_tokens", "mean"),
        avg_reasoning_tokens_persona_mean=("avg_reasoning_tokens", "mean"),
        avg_total_tokens_persona_mean=("avg_total_tokens", "mean"),
    )
)
grouped_descriptive.to_csv(TABLE_DIR / "explicit_implicit_descriptive_six_models.csv", index=False)
display(Markdown("### Ringkasan eksplisit/implisit per identitas"))
display(grouped_descriptive.style.format(precision=3))

# Publication-facing five-category Gender table, following the analytical story
# in the supplied reference while keeping all values in percentage points.
gender_wide = (
    summary[summary["dataset"].eq("Gender")]
    .pivot(index=["model", "model_label"], columns="persona", values="accuracy_pct")
    .reset_index()
)
gender_wide["Man Explicit"] = gender_wide["man_explicit_1"]
gender_wide["Man Implicit Avg"] = gender_wide[[f"man_implicit_{i}" for i in [1, 2, 3]]].mean(axis=1)
gender_wide["Woman Explicit"] = gender_wide["woman_explicit_1"]
gender_wide["Woman Implicit Avg"] = gender_wide[[f"woman_implicit_{i}" for i in [1, 2, 3]]].mean(axis=1)
gender_wide["Neutral"] = gender_wide["neutral"]
gender_wide["Avg Man"] = (gender_wide["Man Explicit"] + gender_wide["Man Implicit Avg"]) / 2
gender_wide["Avg Woman"] = (gender_wide["Woman Explicit"] + gender_wide["Woman Implicit Avg"]) / 2
gender_wide["Delta Man vs Neutral"] = gender_wide["Avg Man"] - gender_wide["Neutral"]
gender_wide["Delta Woman vs Neutral"] = gender_wide["Avg Woman"] - gender_wide["Neutral"]
gender_wide["Gender Gap"] = gender_wide["Avg Man"] - gender_wide["Avg Woman"]
gender_wide["Delta Man Imp-Exp"] = gender_wide["Man Implicit Avg"] - gender_wide["Man Explicit"]
gender_wide["Delta Woman Imp-Exp"] = gender_wide["Woman Implicit Avg"] - gender_wide["Woman Explicit"]
gender_wide["Differential Framing Gap"] = gender_wide["Delta Man Imp-Exp"] - gender_wide["Delta Woman Imp-Exp"]
gender_publication_columns = [
    "model", "model_label", "Neutral", "Man Explicit", "Man Implicit Avg", "Woman Explicit",
    "Woman Implicit Avg", "Avg Man", "Avg Woman", "Delta Man vs Neutral",
    "Delta Woman vs Neutral", "Gender Gap", "Delta Man Imp-Exp", "Delta Woman Imp-Exp",
    "Differential Framing Gap",
]
gender_wide[gender_publication_columns].to_csv(
    TABLE_DIR / "gender_five_category_and_deltas.csv", index=False
)

display(Markdown("### Gender: lima kategori dan delta utama (poin persentase)"))
display(
    gender_wide[[
        "model_label", "Neutral", "Man Explicit", "Man Implicit Avg", "Woman Explicit",
        "Woman Implicit Avg", "Delta Man vs Neutral", "Delta Woman vs Neutral", "Gender Gap",
        "Delta Man Imp-Exp", "Delta Woman Imp-Exp", "Differential Framing Gap",
    ]].style.format(precision=3)
)
"""
    ),
    code(
        r"""
def save_figure(fig: mpl.figure.Figure, stem: str) -> None:
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{stem}.svg", bbox_inches="tight")

model_labels_order = [MODEL_LABELS[model] for model in MODEL_ORDER]
five_categories = ["Man Explicit", "Man Implicit Avg", "Woman Explicit", "Woman Implicit Avg", "Neutral"]
gender_category_labels = [
    "man_explicit", "man_implicit_avg", "woman_explicit", "woman_implicit_avg", "neutral",
]
category_colors = [ACCURACY_COLORS[i] for i in [0, 1, 2, 3, 8]]

# Figure 1 follows the supplied reference: one grouped bar chart, models on X,
# five persona categories side-by-side, and the global Neutral mean as baseline.
plot_gender = gender_wide.copy()
plot_gender["model"] = plot_gender["model"].astype(str)
plot_gender = plot_gender.set_index("model").reindex(MODEL_ORDER).reset_index()
accuracy_plot = plot_gender.set_index("model_label")[five_categories] / 100.0
neutral_baseline = accuracy_plot["Neutral"].mean()
accuracy_plot.columns = gender_category_labels
fig, ax = plt.subplots(figsize=(14.5, 7.2))
accuracy_plot.plot(kind="bar", width=0.84, color=category_colors, ax=ax)
baseline_line = ax.axhline(
    neutral_baseline, color="#D62728", linestyle="--", linewidth=1.2,
    label=f"Global Neutral Avg ({neutral_baseline:.3f})", zorder=1,
)
for bar in ax.patches:
    value = bar.get_height()
    ax.annotate(
        f"{value:.3f}", (bar.get_x() + bar.get_width() / 2, value),
        ha="center", va="bottom", xytext=(0, 4), textcoords="offset points",
        fontsize=7.2, rotation=90,
    )
handles, labels = ax.get_legend_handles_labels()
baseline_index = labels.index(f"Global Neutral Avg ({neutral_baseline:.3f})")
ordered_handles = [handles[baseline_index]] + [h for i, h in enumerate(handles) if i != baseline_index]
ordered_labels = [labels[baseline_index]] + [label for i, label in enumerate(labels) if i != baseline_index]
ax.set_ylim(0.80, 1.00)
ax.set_title("GSM8K Gender: Perbandingan Akurasi Antar-Persona (Skala 0,8–1,0)",
             fontsize=15, fontweight="bold", pad=18)
ax.set_xlabel("Model")
ax.set_ylabel("Accuracy Score")
ax.set_xticklabels(model_labels_order, rotation=32, ha="right")
ax.legend(ordered_handles, ordered_labels, title="Persona & Baseline",
          bbox_to_anchor=(1.01, 1), loc="upper left", frameon=True)
ax.grid(axis="y", linestyle=":", alpha=0.5)
fig.tight_layout()
save_figure(fig, "01_gender_absolute_accuracy_five_categories")
plt.show()

def grouped_delta_bar(data: pd.DataFrame, columns: list[str], labels: list[str], colors: list[str],
                      title: str, stem: str) -> None:
    ordered = data.copy()
    ordered["model"] = ordered["model"].astype(str)
    ordered = ordered.set_index("model").reindex(MODEL_ORDER).reset_index()
    # The reference reports deltas on the 0–1 accuracy scale, hence /100 here.
    delta_plot = ordered.set_index("model_label")[columns] / 100.0
    delta_plot.columns = labels
    fig, ax = plt.subplots(figsize=(14.5, 6.8))
    delta_plot.plot(kind="bar", width=0.78, color=colors, ax=ax)
    ax.axhline(0, color="black", linewidth=1)
    for bar in ax.patches:
        value = bar.get_height()
        ax.annotate(
            f"{value:.4f}", (bar.get_x() + bar.get_width() / 2, value),
            ha="center", va="bottom" if value >= 0 else "top",
            xytext=(0, 4 if value >= 0 else -4), textcoords="offset points", fontsize=7.5,
        )
    max_abs = max(0.001, np.nanmax(np.abs(delta_plot.to_numpy())))
    ax.set_ylim(-max_abs * 1.38, max_abs * 1.38)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=16)
    ax.set_xlabel("Model")
    ax.set_ylabel("Delta Value")
    ax.set_xticklabels(model_labels_order, rotation=32, ha="right")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, stem)
    plt.show()

# Figure 2: external deviation from the neutral baseline.
grouped_delta_bar(
    plot_gender,
    ["Delta Man vs Neutral", "Delta Woman vs Neutral", "Gender Gap"],
    ["man_avg - neutral", "woman_avg - neutral", "man_avg - woman_avg"],
    GENDER_DELTA_COLORS,
    "GSM8K Gender: Analisis Delta terhadap Persona Neutral",
    "02_gender_delta_vs_neutral",
)

# Figure 3: internal prompt-framing sensitivity, the second delta story in the reference.
grouped_delta_bar(
    plot_gender,
    ["Delta Man Imp-Exp", "Delta Woman Imp-Exp", "Gender Gap"],
    ["man_implicit_avg - man_explicit", "woman_implicit_avg - woman_explicit", "man_avg - woman_avg"],
    GENDER_DELTA_COLORS,
    "GSM8K Gender — Analisis Delta: Performance Gap dan Gender Bias",
    "03_gender_delta_implicit_vs_explicit",
)
"""
    ),
    md(
        r"""
## 3. Effect size akurasi

Semua arah ditulis sebagai **A − B**:

- `risk_difference_pp`: selisih akurasi dalam poin persentase;
- `cohen_h`: standardized difference untuk dua proporsi marginal;
- `marginal_odds_ratio`: odds benar A dibagi odds benar B, dengan koreksi 0,5 bila ada sel nol.

Pedoman Cohen's h yang lazim (`|h|≈0,2/0,5/0,8`) hanya konvensi, bukan batas substantif otomatis. Karena data tingkat pertanyaan tidak ada di dua CSV, tabel ini **deskriptif** dan tidak memuat p-value yang mengasumsikan independensi.
"""
    ),
    code(
        r"""
def build_comparisons() -> pd.DataFrame:
    rows = []
    for framing, suffix in [("explicit", "_explicit_1"), ("implicit 1", "_implicit_1"),
                            ("implicit 2", "_implicit_2"), ("implicit 3", "_implicit_3")]:
        rows.append({"dataset": "Gender", "family": "gender_matched",
                     "label": f"Man − Woman | {framing}",
                     "persona_a": f"man{suffix}", "persona_b": f"woman{suffix}"})
    for identity in ["man", "woman"]:
        for framing, suffix in [("explicit", "explicit_1"), ("implicit 1", "implicit_1"),
                                ("implicit 2", "implicit_2"), ("implicit 3", "implicit_3")]:
            rows.append({"dataset": "Gender", "family": "neutral_baseline",
                         "label": f"{identity.title()} {framing} − Neutral",
                         "persona_a": f"{identity}_{suffix}", "persona_b": "neutral"})
    for identity in ["man", "woman"]:
        for variant in [1, 2, 3]:
            rows.append({"dataset": "Gender", "family": "framing_within_identity",
                         "label": f"{identity.title()} implicit {variant} − explicit",
                         "persona_a": f"{identity}_implicit_{variant}", "persona_b": f"{identity}_explicit_1"})
    identities = ["atheist", "christian", "islam", "jewish"]
    for identity_a, identity_b in itertools.combinations(identities, 2):
        for framing, suffix in [("explicit", "explicit"), ("implicit 1", "implicit_1"),
                                ("implicit 2", "implicit_2"), ("implicit 3", "implicit_3")]:
            rows.append({"dataset": "Religion", "family": "religion_identity_matched",
                         "label": f"{identity_a.title()} − {identity_b.title()} | {framing}",
                         "persona_a": f"{identity_a}_{suffix}", "persona_b": f"{identity_b}_{suffix}"})
    for identity in identities:
        for variant in [1, 2, 3]:
            rows.append({"dataset": "Religion", "family": "framing_within_religion",
                         "label": f"{identity.title()} implicit {variant} − explicit",
                         "persona_a": f"{identity}_implicit_{variant}", "persona_b": f"{identity}_explicit"})
    return pd.DataFrame(rows)

comparisons = build_comparisons()

def marginal_effect(row_a: pd.Series, row_b: pd.Series) -> dict:
    correct_a, total_a = float(row_a["correct"]), float(row_a["total"])
    correct_b, total_b = float(row_b["correct"]), float(row_b["total"])
    p_a, p_b = float(row_a["accuracy_pct"]) / 100, float(row_b["accuracy_pct"]) / 100
    odds_a = p_a / (1 - p_a) if p_a < 1 else np.inf
    odds_b = p_b / (1 - p_b) if p_b < 1 else np.inf
    odds_ratio = odds_a / odds_b
    return {
        "reported_correct_a": int(correct_a), "total_a": int(total_a),
        "reported_correct_b": int(correct_b), "total_b": int(total_b),
        "accuracy_a_pct": 100 * p_a, "accuracy_b_pct": 100 * p_b,
        "risk_difference_pp": 100 * (p_a - p_b),
        "cohen_h": 2 * math.asin(math.sqrt(p_a)) - 2 * math.asin(math.sqrt(p_b)),
        "marginal_odds_ratio": odds_ratio,
    }

effect_rows = []
for (dataset, model), model_frame in summary.groupby(["dataset", "model"], observed=True):
    indexed = model_frame.set_index("persona")
    for comparison in comparisons[comparisons["dataset"].eq(dataset)].to_dict("records"):
        if comparison["persona_a"] not in indexed.index or comparison["persona_b"] not in indexed.index:
            continue
        effect_rows.append({
            **comparison,
            "model": str(model),
            "model_label": MODEL_LABELS[str(model)],
            **marginal_effect(indexed.loc[comparison["persona_a"]], indexed.loc[comparison["persona_b"]]),
            "inference_status": "descriptive_only_aggregate_source",
        })
accuracy_effects = pd.DataFrame(effect_rows)
accuracy_effects["abs_cohen_h"] = accuracy_effects["cohen_h"].abs()
accuracy_effects["cohen_h_magnitude"] = pd.cut(
    accuracy_effects["abs_cohen_h"],
    bins=[-np.inf, 0.2, 0.5, 0.8, np.inf],
    labels=["negligible", "small", "medium", "large"],
    right=False,
)
accuracy_effects.to_csv(TABLE_DIR / "marginal_accuracy_effects_six_models.csv", index=False)

display(
    accuracy_effects.sort_values("risk_difference_pp", key=lambda values: values.abs(), ascending=False)
    [["dataset", "model_label", "family", "label", "accuracy_a_pct", "accuracy_b_pct",
      "risk_difference_pp", "cohen_h", "cohen_h_magnitude", "marginal_odds_ratio"]]
    .head(25).style.format(precision=4)
)
"""
    ),
    code(
        r"""
# Figure 4: granular effect-size map, corresponding to the most useful reference
# heatmap but omitting invalid aggregate-source p-values.
gender_effect_order = [
    "Man − Woman | explicit", "Man − Woman | implicit 1", "Man − Woman | implicit 2",
    "Man − Woman | implicit 3",
    "Man implicit 1 − explicit", "Man implicit 2 − explicit", "Man implicit 3 − explicit",
    "Woman implicit 1 − explicit", "Woman implicit 2 − explicit", "Woman implicit 3 − explicit",
    "Man explicit − Neutral", "Woman explicit − Neutral",
    "Man implicit 1 − Neutral", "Man implicit 2 − Neutral", "Man implicit 3 − Neutral",
    "Woman implicit 1 − Neutral", "Woman implicit 2 − Neutral", "Woman implicit 3 − Neutral",
]
gender_effect_matrix = (
    accuracy_effects[accuracy_effects["dataset"].eq("Gender")]
    .pivot(index="model_label", columns="label", values="cohen_h")
    .reindex(index=model_labels_order, columns=gender_effect_order)
)
gender_effect_matrix.columns = [
    "man_explicit - woman_explicit",
    "man_implicit_1 - woman_implicit_1", "man_implicit_2 - woman_implicit_2",
    "man_implicit_3 - woman_implicit_3",
    "man_implicit_1 - man_explicit", "man_implicit_2 - man_explicit",
    "man_implicit_3 - man_explicit",
    "woman_implicit_1 - woman_explicit", "woman_implicit_2 - woman_explicit",
    "woman_implicit_3 - woman_explicit",
    "man_explicit - neutral", "woman_explicit - neutral",
    "man_implicit_1 - neutral", "man_implicit_2 - neutral", "man_implicit_3 - neutral",
    "woman_implicit_1 - neutral", "woman_implicit_2 - neutral", "woman_implicit_3 - neutral",
]
fig, ax = plt.subplots(figsize=(19, 5.6))
limit = np.nanmax(np.abs(gender_effect_matrix.to_numpy()))
sns.heatmap(gender_effect_matrix, annot=True, fmt="+.02f", center=0, vmin=-limit, vmax=limit,
            cmap=EFFECT_CMAP, linewidths=0.4, cbar_kws={"label": "Cohen's h (A − B)"}, ax=ax)
ax.set_title("GSM8K Gender: Effect Size untuk Setiap Perbandingan Kondisi")
ax.set_xlabel("Perbandingan (arah A − B)")
ax.set_ylabel("Model")
ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
save_figure(fig, "04_gender_granular_effect_size_heatmap")
plt.show()

# Religion absolute level: explicit and mean implicit should be visible together;
# centering alone would hide whether a delta occurs at high or low accuracy.
religion_raw = summary[summary["dataset"].eq("Religion")].copy()
religion_wide = religion_raw.pivot(index=["model", "model_label"], columns="persona", values="accuracy_pct").reset_index()
religion_identities = ["atheist", "christian", "islam", "jewish"]
for identity in religion_identities:
    religion_wide[f"{identity.title()} Explicit"] = religion_wide[f"{identity}_explicit"]
    religion_wide[f"{identity.title()} Implicit Avg"] = religion_wide[[f"{identity}_implicit_{i}" for i in [1, 2, 3]]].mean(axis=1)
    religion_wide[f"{identity.title()} Delta Imp-Exp"] = (
        religion_wide[f"{identity.title()} Implicit Avg"] - religion_wide[f"{identity.title()} Explicit"]
    )
    religion_wide[f"{identity.title()} Overall Avg"] = (
        religion_wide[f"{identity.title()} Explicit"] + religion_wide[f"{identity.title()} Implicit Avg"]
    ) / 2
# Religion CSV has no Neutral row. Use the matching model's Neutral accuracy
# from the Gender & Neutral CSV so the Religion chart has the same control
# category and global baseline as the supplied reference.
gender_neutral_lookup = dict(zip(gender_wide["model"].astype(str), gender_wide["Neutral"]))
religion_wide["Neutral"] = religion_wide["model"].astype(str).map(gender_neutral_lookup)
assert religion_wide["Neutral"].notna().all()
religion_publication_columns = ["model", "model_label", "Neutral"]
for identity in religion_identities:
    religion_publication_columns.extend([
        f"{identity.title()} Explicit", f"{identity.title()} Implicit Avg",
        f"{identity.title()} Delta Imp-Exp", f"{identity.title()} Overall Avg",
    ])
religion_wide[religion_publication_columns].to_csv(
    TABLE_DIR / "religion_explicit_implicit_and_deltas.csv", index=False
)

religion_plot = religion_wide.copy()
religion_plot["model"] = religion_plot["model"].astype(str)
religion_plot = religion_plot.set_index("model").reindex(MODEL_ORDER).reset_index()
religion_accuracy_columns = [
    column
    for identity in religion_identities
    for column in [f"{identity.title()} Explicit", f"{identity.title()} Implicit Avg"]
] + ["Neutral"]
religion_accuracy_colors = ACCURACY_COLORS
religion_accuracy_plot = religion_plot.set_index("model_label")[religion_accuracy_columns] / 100.0
religion_neutral_baseline = religion_accuracy_plot["Neutral"].mean()
religion_accuracy_plot.columns = [
    column for identity in religion_identities
    for column in [f"{identity}_explicit", f"{identity}_implicit_avg"]
] + ["neutral"]
fig, ax = plt.subplots(figsize=(16.5, 7.4))
religion_accuracy_plot.plot(kind="bar", width=0.88, color=religion_accuracy_colors, ax=ax)
ax.axhline(
    religion_neutral_baseline, color="#D62728", linestyle="--", linewidth=1.2,
    label=f"Global Neutral Avg ({religion_neutral_baseline:.3f})", zorder=1,
)
for bar in ax.patches:
    value = bar.get_height()
    ax.annotate(
        f"{value:.3f}", (bar.get_x() + bar.get_width() / 2, value),
        ha="center", va="bottom", xytext=(0, 3), textcoords="offset points",
        fontsize=6.2, rotation=90,
    )
ax.set_ylim(0.75, 1.00)
ax.set_title("GSM8K Religion: Perbandingan Akurasi Antar-Persona (Skala 0,75–1,0)",
             fontsize=15, fontweight="bold", pad=18)
ax.set_xlabel("Model")
ax.set_ylabel("Accuracy Score")
ax.set_xticklabels(model_labels_order, rotation=32, ha="right")
religion_handles, religion_labels = ax.get_legend_handles_labels()
religion_baseline_label = f"Global Neutral Avg ({religion_neutral_baseline:.3f})"
religion_baseline_index = religion_labels.index(religion_baseline_label)
religion_handles = [religion_handles[religion_baseline_index]] + [
    handle for index, handle in enumerate(religion_handles) if index != religion_baseline_index
]
religion_labels = [religion_baseline_label] + [
    label for index, label in enumerate(religion_labels) if index != religion_baseline_index
]
ax.legend(religion_handles, religion_labels, title="Religion Persona & Baseline",
          bbox_to_anchor=(1.01, 1), loc="upper left", frameon=True)
ax.grid(axis="y", linestyle=":", alpha=0.5)
fig.tight_layout()
save_figure(fig, "05_religion_absolute_accuracy_explicit_implicit")
plt.show()

# Figure 6 follows the supplied delta reference: models on X, one bar per
# identity, numeric labels, and a horizontal zero baseline.
religion_delta_columns = [f"{identity.title()} Delta Imp-Exp" for identity in religion_identities]
religion_delta_plot = religion_plot.set_index("model_label")[religion_delta_columns] / 100.0
religion_delta_plot.columns = [
    f"{identity}_implicit_avg - {identity}_explicit" for identity in religion_identities
]
fig, ax = plt.subplots(figsize=(15.5, 6.8))
religion_delta_plot.plot(kind="bar", width=0.82, color=DELTA_COLORS, ax=ax)
ax.axhline(0, color="black", linewidth=1)
for bar in ax.patches:
    value = bar.get_height()
    ax.annotate(
        f"{value:.4f}", (bar.get_x() + bar.get_width() / 2, value),
        ha="center", va="bottom" if value >= 0 else "top",
        xytext=(0, 4 if value >= 0 else -4), textcoords="offset points", fontsize=7.2,
    )
religion_delta_limit = max(0.001, np.nanmax(np.abs(religion_delta_plot.to_numpy())))
ax.set_ylim(-religion_delta_limit * 1.35, religion_delta_limit * 1.35)
ax.set_title("GSM8K Religion — Analisis Delta: Persona Implicit vs Explicit",
             fontsize=15, fontweight="bold", pad=16)
ax.set_xlabel("Model")
ax.set_ylabel("Delta Value")
ax.set_xticklabels(model_labels_order, rotation=32, ha="right")
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=True)
ax.grid(axis="y", linestyle="--", alpha=0.3)
fig.tight_layout()
save_figure(fig, "06_religion_delta_implicit_vs_explicit")
plt.show()

# Figure 7: pairwise identity effect after equal-weight aggregation across the
# explicit and mean-implicit conditions.
identity_effect_rows = []
for _, row in religion_plot.iterrows():
    for identity_a, identity_b in itertools.combinations(religion_identities, 2):
        p_a = row[f"{identity_a.title()} Overall Avg"] / 100
        p_b = row[f"{identity_b.title()} Overall Avg"] / 100
        identity_effect_rows.append({
            "model": row["model"], "model_label": row["model_label"],
            "comparison": f"{identity_a} - {identity_b}",
            "accuracy_a_pct": 100 * p_a, "accuracy_b_pct": 100 * p_b,
            "risk_difference_pp": 100 * (p_a - p_b),
            "cohen_h": 2 * math.asin(math.sqrt(p_a)) - 2 * math.asin(math.sqrt(p_b)),
        })
religion_identity_effects = pd.DataFrame(identity_effect_rows)
religion_identity_effects.to_csv(TABLE_DIR / "religion_identity_pair_effects.csv", index=False)
identity_effect_matrix = (
    religion_identity_effects.pivot(index="model_label", columns="comparison", values="cohen_h")
    .reindex(index=model_labels_order)
)
fig, ax = plt.subplots(figsize=(11.5, 5.3))
effect_limit = max(0.01, np.nanmax(np.abs(identity_effect_matrix.to_numpy())))
sns.heatmap(identity_effect_matrix, annot=True, fmt="+.02f", center=0, vmin=-effect_limit, vmax=effect_limit,
            cmap=EFFECT_CMAP, linewidths=0.5, cbar_kws={"label": "Cohen's h (A − B)"}, ax=ax)
ax.set_title("GSM8K Religion: Effect Size Antar-Identitas")
ax.set_xlabel("Perbandingan identitas")
ax.set_ylabel("Model")
ax.tick_params(axis="x", rotation=35)
fig.tight_layout()
save_figure(fig, "07_religion_identity_effect_size_heatmap")
plt.show()
"""
    ),
    md(
        r"""
## 4. Analisis token

Analisis mempertahankan prompt, completion, reasoning, dan total token. Karena sumber hanya berisi agregat:

- `token_difference`: rerata token A dikurangi B;
- `token_ratio_a_over_b`: rasio rerata token A/B;
- tidak ada standardized effect atau p-value tanpa simpangan baku/data tingkat respons.

Token antarmodel tidak sepenuhnya sebanding karena tokenizer dan pelaporan reasoning dapat berbeda. Fokus inferensi substantif sebaiknya pada perbandingan persona **di dalam model**.
"""
    ),
    code(
        r"""
TOKEN_METRICS = ["avg_prompt_tokens", "avg_completion_tokens", "avg_reasoning_tokens", "avg_total_tokens"]
token_effect_rows = []
for (dataset, model), model_frame in summary.groupby(["dataset", "model"], observed=True):
    indexed = model_frame.set_index("persona")
    for comparison in comparisons[comparisons["dataset"].eq(dataset)].to_dict("records"):
        if comparison["persona_a"] not in indexed.index or comparison["persona_b"] not in indexed.index:
            continue
        row_a, row_b = indexed.loc[comparison["persona_a"]], indexed.loc[comparison["persona_b"]]
        for metric in TOKEN_METRICS:
            value_a, value_b = row_a[metric], row_b[metric]
            token_effect_rows.append({
                **comparison,
                "model": str(model), "model_label": MODEL_LABELS[str(model)],
                "token_metric": metric,
                "value_a": value_a, "value_b": value_b,
                "token_difference": value_a - value_b if pd.notna(value_a) and pd.notna(value_b) else np.nan,
                "token_ratio_a_over_b": value_a / value_b if pd.notna(value_a) and pd.notna(value_b) and value_b != 0 else np.nan,
                "inference_status": "descriptive_only_aggregate_source",
            })
token_effects = pd.DataFrame(token_effect_rows)
token_effects.to_csv(TABLE_DIR / "token_descriptive_effects_six_models.csv", index=False)

model_efficiency = (
    summary.groupby(["dataset", "model", "model_label"], observed=True, as_index=False)
    .agg(
        personas=("persona", "nunique"),
        responses=("total", "sum"),
        correct=("correct", "sum"),
        prompt_tokens_sum=("sum_prompt_tokens", lambda values: values.sum(min_count=1)),
        completion_tokens_sum=("sum_completion_tokens", lambda values: values.sum(min_count=1)),
        reasoning_tokens_sum=("sum_reasoning_tokens", lambda values: values.sum(min_count=1)),
        total_tokens_sum=("sum_total_tokens", lambda values: values.sum(min_count=1)),
        prompt_tokens_persona_mean=("avg_prompt_tokens", "mean"),
        completion_tokens_persona_mean=("avg_completion_tokens", "mean"),
        reasoning_tokens_persona_mean=("avg_reasoning_tokens", "mean"),
        total_tokens_persona_mean=("avg_total_tokens", "mean"),
        token_personas_available=("avg_total_tokens", "count"),
    )
)
weighted_model_accuracy = (
    summary.assign(weighted_accuracy=lambda frame: frame["accuracy_pct"] * frame["total"])
    .groupby(["dataset", "model"], observed=True)["weighted_accuracy"].sum()
    / summary.groupby(["dataset", "model"], observed=True)["total"].sum()
).rename("accuracy_pooled_pct").reset_index()
model_efficiency = model_efficiency.merge(
    weighted_model_accuracy, on=["dataset", "model"], validate="one_to_one"
)
model_efficiency["accuracy_implied_correct"] = (
    model_efficiency["accuracy_pooled_pct"] * model_efficiency["responses"] / 100
)
model_efficiency["correct_per_1k_completion_tokens"] = 1_000 * model_efficiency["accuracy_implied_correct"] / model_efficiency["completion_tokens_sum"]
model_efficiency["correct_per_1k_total_tokens"] = 1_000 * model_efficiency["accuracy_implied_correct"] / model_efficiency["total_tokens_sum"]
model_efficiency.to_csv(TABLE_DIR / "model_token_efficiency_six_models.csv", index=False)

display(Markdown("### Perbedaan completion token terbesar"))
display(
    token_effects[token_effects["token_metric"].eq("avg_completion_tokens")]
    .sort_values("token_difference", key=lambda values: values.abs(), ascending=False)
    [["dataset", "model_label", "label", "value_a", "value_b", "token_difference", "token_ratio_a_over_b"]]
    .head(25).style.format(precision=3)
)
display(Markdown("### Efisiensi token per model-dataset"))
display(model_efficiency.style.format(precision=3))
"""
    ),
    code(
        r"""
def aggregate_token_categories(dataset: str) -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        model_frame = summary[summary["dataset"].eq(dataset) & summary["model"].astype(str).eq(model)].set_index("persona")
        if dataset == "Gender":
            definitions = {
                "Neutral": ["neutral"],
                "Man Explicit": ["man_explicit_1"],
                "Man Implicit Avg": [f"man_implicit_{i}" for i in [1, 2, 3]],
                "Woman Explicit": ["woman_explicit_1"],
                "Woman Implicit Avg": [f"woman_implicit_{i}" for i in [1, 2, 3]],
            }
        else:
            definitions = {}
            for identity in religion_identities:
                definitions[f"{identity.title()} Explicit"] = [f"{identity}_explicit"]
                definitions[f"{identity.title()} Implicit Avg"] = [f"{identity}_implicit_{i}" for i in [1, 2, 3]]
        for category, personas in definitions.items():
            record = {"dataset": dataset, "model": model, "model_label": MODEL_LABELS[model], "category": category}
            for metric in TOKEN_METRICS:
                record[metric] = model_frame.loc[personas, metric].mean()
            rows.append(record)
    return pd.DataFrame(rows)

gender_token_categories = aggregate_token_categories("Gender")
religion_token_categories = aggregate_token_categories("Religion")
token_category_summary = pd.concat([gender_token_categories, religion_token_categories], ignore_index=True)
token_category_summary.to_csv(TABLE_DIR / "token_category_summary_six_models.csv", index=False)

def token_matrix(frame: pd.DataFrame, metric: str, categories: list[str]) -> pd.DataFrame:
    matrix = (
        frame.pivot(index="model_label", columns="category", values=metric)
        .reindex(index=model_labels_order, columns=categories)
    )
    # Seaborn cannot cast pandas.NA from mixed nullable dtypes after Regional
    # rows are concatenated. Preserve missingness as ordinary floating NaN.
    return matrix.apply(pd.to_numeric, errors="coerce").astype(float)

category_axis_labels = {
    "Neutral": "neutral",
    "Man Explicit": "man_explicit",
    "Man Implicit Avg": "man_implicit_avg",
    "Woman Explicit": "woman_explicit",
    "Woman Implicit Avg": "woman_implicit_avg",
}
for identity in religion_identities:
    category_axis_labels[f"{identity.title()} Explicit"] = f"{identity}_explicit"
    category_axis_labels[f"{identity.title()} Implicit Avg"] = f"{identity}_implicit_avg"

# Figure 8: the reader can compare token components within a model and persona
# category. Components are separate panels, not stacked, because reasoning may
# already be included in a provider's completion count.
fig, axes = plt.subplots(1, 3, figsize=(19, 5.6), constrained_layout=True)
for ax, metric, title in zip(
    axes,
    ["avg_prompt_tokens", "avg_completion_tokens", "avg_total_tokens"],
    ["Token Masukan", "Token Keluaran", "Total Token"],
):
    matrix = token_matrix(gender_token_categories, metric, five_categories)
    matrix.columns = [category_axis_labels[column] for column in matrix.columns]
    sns.heatmap(matrix, annot=True, fmt=".0f", cmap=TOKEN_CMAP, linewidths=0.4,
                cbar_kws={"label": "Rerata token"}, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Model" if ax is axes[0] else "")
    ax.tick_params(axis="x", rotation=35)
fig.suptitle("GSM8K Gender: Penggunaan Token per Kategori Persona", fontsize=15, fontweight="bold")
save_figure(fig, "08_gender_token_components_by_category")
plt.show()

# Figure 9: persona overhead relative to Neutral is more interpretable than a
# cross-model token ranking. Ministral Neutral is absent in the source and stays blank.
gender_total = token_matrix(gender_token_categories, "avg_total_tokens", five_categories)
gender_token_delta = gender_total.drop(columns="Neutral").sub(gender_total["Neutral"], axis=0)
gender_token_delta_pct = gender_token_delta.div(gender_total["Neutral"], axis=0) * 100
gender_token_delta_pct.columns = [category_axis_labels[column] for column in gender_token_delta_pct.columns]
gender_token_delta_pct.to_csv(TABLE_DIR / "gender_token_delta_vs_neutral_pct.csv")
fig, ax = plt.subplots(figsize=(10.5, 5.3))
token_limit = max(1.0, np.nanmax(np.abs(gender_token_delta_pct.to_numpy())))
gender_token_annotations = gender_token_delta_pct.map(
    lambda value: "" if pd.isna(value) else ("0" if abs(value) < 0.5 else f"{value:+.0f}")
)
sns.heatmap(gender_token_delta_pct, annot=gender_token_annotations, fmt="", center=0,
            vmin=-token_limit, vmax=token_limit,
            cmap=TOKEN_DELTA_CMAP, linewidths=0.5,
            cbar_kws={"label": "Perubahan vs Neutral (%)"}, ax=ax)
for row_index, column_index in np.argwhere(gender_token_delta_pct.isna().to_numpy()):
    ax.text(column_index + 0.5, row_index + 0.5, "N/A", ha="center", va="center",
            fontsize=9, color="#666666", fontstyle="italic")
ax.set_title("GSM8K Gender: Total Token Overhead terhadap Persona Neutral")
ax.set_xlabel("Kategori persona")
ax.set_ylabel("Model")
ax.tick_params(axis="x", rotation=25)
fig.tight_layout()
save_figure(fig, "09_gender_total_token_delta_vs_neutral")
plt.show()

# Figure 10: Religion has no neutral baseline; show explicit and implicit mean
# side-by-side for every identity, with all four reported token fields.
religion_token_order = [
    category for identity in religion_identities
    for category in [f"{identity.title()} Explicit", f"{identity.title()} Implicit Avg"]
]
fig, axes = plt.subplots(2, 2, figsize=(18.5, 10.2), constrained_layout=True)
for ax, metric, title in zip(
    axes.flat,
    ["avg_prompt_tokens", "avg_completion_tokens", "avg_reasoning_tokens", "avg_total_tokens"],
    ["Token Masukan", "Token Keluaran", "Token Penalaran", "Total Token"],
):
    matrix = token_matrix(religion_token_categories, metric, religion_token_order)
    matrix.columns = [category_axis_labels[column] for column in matrix.columns]
    sns.heatmap(matrix, annot=True, fmt=".0f", cmap=TOKEN_CMAP, linewidths=0.35,
                cbar_kws={"label": "Rerata token"}, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Model")
    ax.tick_params(axis="x", rotation=35)
fig.suptitle("GSM8K Religion: Penggunaan Token Persona Explicit dan Rerata Implicit",
             fontsize=15, fontweight="bold")
save_figure(fig, "10_religion_token_components_explicit_implicit")
plt.show()

# Figure 11: within-identity token framing delta in percent.
religion_total_tokens = token_matrix(religion_token_categories, "avg_total_tokens", religion_token_order)
religion_token_delta_pct = pd.DataFrame(index=model_labels_order)
for identity in religion_identities:
    explicit = religion_total_tokens[f"{identity.title()} Explicit"]
    implicit = religion_total_tokens[f"{identity.title()} Implicit Avg"]
    religion_token_delta_pct[f"{identity}_implicit_avg - {identity}_explicit"] = (
        100 * (implicit - explicit) / explicit
    )
religion_token_delta_pct.to_csv(TABLE_DIR / "religion_token_delta_implicit_vs_explicit_pct.csv")
fig, ax = plt.subplots(figsize=(8.8, 5.3))
religion_token_limit = max(1.0, np.nanmax(np.abs(religion_token_delta_pct.to_numpy())))
sns.heatmap(religion_token_delta_pct, annot=True, fmt="+.0f", center=0,
            vmin=-religion_token_limit, vmax=religion_token_limit,
            cmap=TOKEN_DELTA_CMAP, linewidths=0.5,
            cbar_kws={"label": "Perubahan Implicit vs Explicit (%)"}, ax=ax)
ax.set_title("GSM8K Religion: Total Token Overhead Persona Implicit terhadap Explicit")
ax.set_xlabel("Identitas persona")
ax.set_ylabel("Model")
fig.tight_layout()
save_figure(fig, "11_religion_total_token_delta_implicit_vs_explicit")
plt.show()
"""
    ),
    md(
        r"""
## 5. Regional Cultural: akurasi, delta, effect size, significance, dan token

Regional Cultural memiliki empat identitas wilayah (`North American`, `East Asian`,
`South Asian`, dan `Sub-Saharan African`), masing-masing satu prompt Explicit dan tiga
varian Implicit. Grafik level memakai rerata tiga varian Implicit agar pembaca dapat
membandingkan delapan kelompok utama tanpa kehilangan tabel granularnya.

Untuk setiap model-region-varian, outcome Implicit dan Explicit disejajarkan berdasarkan
`question_id`. Significance value dihitung dengan **exact McNemar** dari pasangan discordant,
sedangkan besar efek dilaporkan sebagai risk difference, Cohen's h, dan matched odds ratio.
Seluruh 72 p-value dikoreksi bersama dengan Benjamini-Hochberg FDR; kolom `q_value_bh_fdr`
adalah significance value utama. Abstain/missing tetap dihitung tidak benar pada outcome
utama agar konsisten dengan headline accuracy.

Neutral pada grafik Regional berasal dari baris Neutral Gender untuk model yang sama. Ini
hanya kontrol visual deskriptif; notebook tidak menghitung p-value Regional-vs-Neutral karena
data pasangan Neutral tingkat soal tidak tersedia di sumber Regional.
"""
    ),
    code(
        r"""
REGION_LABELS = {
    "north_american": "North American",
    "east_asian": "East Asian",
    "south_asian": "South Asian",
    "sub_saharan_african": "Sub-Saharan African",
}

regional_wide = (
    summary[summary["dataset"].eq("Regional Cultural")]
    .pivot(index=["model", "model_label"], columns="persona", values="accuracy_pct")
    .reset_index()
)
for identity in REGIONAL_IDENTITIES:
    label = REGION_LABELS[identity]
    regional_wide[f"{label} Explicit"] = regional_wide[f"{identity}_explicit"]
    regional_wide[f"{label} Implicit Avg"] = regional_wide[
        [f"{identity}_implicit_{variant}" for variant in [1, 2, 3]]
    ].mean(axis=1)
    regional_wide[f"{label} Delta Imp-Exp"] = (
        regional_wide[f"{label} Implicit Avg"] - regional_wide[f"{label} Explicit"]
    )
regional_wide["Neutral"] = regional_wide["model"].astype(str).map(gender_neutral_lookup)
assert regional_wide["Neutral"].notna().all()

regional_table_columns = ["model", "model_label", "Neutral"]
for identity in REGIONAL_IDENTITIES:
    label = REGION_LABELS[identity]
    regional_table_columns.extend([
        f"{label} Explicit", f"{label} Implicit Avg", f"{label} Delta Imp-Exp",
    ])
regional_wide[regional_table_columns].to_csv(TABLE_DIR / "regional_accuracy_and_deltas.csv", index=False)

regional_plot = regional_wide.copy()
regional_plot["model"] = regional_plot["model"].astype(str)
regional_plot = regional_plot.set_index("model").reindex(MODEL_ORDER).reset_index()
regional_accuracy_columns = [
    column for identity in REGIONAL_IDENTITIES
    for column in [
        f"{REGION_LABELS[identity]} Explicit", f"{REGION_LABELS[identity]} Implicit Avg",
    ]
] + ["Neutral"]
regional_accuracy_colors = ACCURACY_COLORS
regional_accuracy_plot = regional_plot.set_index("model_label")[regional_accuracy_columns] / 100
regional_neutral_baseline = regional_accuracy_plot["Neutral"].mean()
regional_accuracy_plot.columns = [
    column for identity in REGIONAL_IDENTITIES
    for column in [f"{identity}_explicit", f"{identity}_implicit_avg"]
] + ["neutral"]

# Figure 12: grouped bars plus the same Global Neutral Average visual baseline.
fig, ax = plt.subplots(figsize=(17, 7.5))
regional_accuracy_plot.plot(kind="bar", width=0.88, color=regional_accuracy_colors, ax=ax)
regional_baseline_label = f"Global Neutral Avg ({regional_neutral_baseline:.3f})"
ax.axhline(regional_neutral_baseline, color="#D62728", linestyle="--", linewidth=1.2,
           label=regional_baseline_label, zorder=1)
for bar in ax.patches:
    value = bar.get_height()
    ax.annotate(f"{value:.3f}", (bar.get_x() + bar.get_width() / 2, value),
                ha="center", va="bottom", xytext=(0, 3), textcoords="offset points",
                fontsize=6.1, rotation=90)
regional_min = max(0.70, math.floor((np.nanmin(regional_accuracy_plot.to_numpy()) - 0.02) * 20) / 20)
ax.set_ylim(regional_min, 1.00)
ax.set_title("GSM8K Regional Cultural: Perbandingan Akurasi Antar-Persona",
             fontsize=15, fontweight="bold", pad=18)
ax.set_xlabel("Model")
ax.set_ylabel("Accuracy Score")
ax.set_xticklabels(model_labels_order, rotation=32, ha="right")
handles, labels = ax.get_legend_handles_labels()
baseline_index = labels.index(regional_baseline_label)
handles = [handles[baseline_index]] + [h for i, h in enumerate(handles) if i != baseline_index]
labels = [regional_baseline_label] + [label for i, label in enumerate(labels) if i != baseline_index]
ax.legend(handles, labels, title="Regional Persona & Baseline",
          bbox_to_anchor=(1.01, 1), loc="upper left", frameon=True)
ax.grid(axis="y", linestyle=":", alpha=0.5)
fig.tight_layout()
save_figure(fig, "12_regional_absolute_accuracy_explicit_implicit")
plt.show()

# Figure 13: delta Implicit Avg - Explicit, one bar per region and model.
regional_delta_columns = [f"{REGION_LABELS[identity]} Delta Imp-Exp" for identity in REGIONAL_IDENTITIES]
regional_delta_plot = regional_plot.set_index("model_label")[regional_delta_columns] / 100
regional_delta_plot.columns = [
    f"{identity}_implicit_avg - {identity}_explicit" for identity in REGIONAL_IDENTITIES
]
fig, ax = plt.subplots(figsize=(15.5, 6.8))
regional_delta_plot.plot(kind="bar", width=0.82, color=DELTA_COLORS, ax=ax)
ax.axhline(0, color="black", linewidth=1)
for bar in ax.patches:
    value = bar.get_height()
    ax.annotate(f"{value:.4f}", (bar.get_x() + bar.get_width() / 2, value),
                ha="center", va="bottom" if value >= 0 else "top",
                xytext=(0, 4 if value >= 0 else -4), textcoords="offset points", fontsize=7.2)
delta_limit = max(0.001, np.nanmax(np.abs(regional_delta_plot.to_numpy())))
ax.set_ylim(-delta_limit * 1.4, delta_limit * 1.4)
ax.set_title("GSM8K Regional Cultural — Analisis Delta: Persona Implicit vs Explicit",
             fontsize=15, fontweight="bold", pad=16)
ax.set_xlabel("Model")
ax.set_ylabel("Delta Value")
ax.set_xticklabels(model_labels_order, rotation=32, ha="right")
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=True)
ax.grid(axis="y", linestyle="--", alpha=0.3)
fig.tight_layout()
save_figure(fig, "13_regional_delta_implicit_vs_explicit")
plt.show()
"""
    ),
    code(
        r"""
def bh_fdr(p_values: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    result = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    clean = values[valid]
    order = np.argsort(clean)
    ranked = clean[order]
    adjusted_ranked = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0, 1)
    restored = np.empty_like(adjusted_ranked)
    restored[order] = adjusted_ranked
    result[valid] = restored
    return result

paired_rows = []
for model in MODEL_ORDER:
    model_detail = regional_detail[regional_detail["model"].eq(model)]
    for identity in REGIONAL_IDENTITIES:
        explicit = (
            model_detail[model_detail["persona"].eq(f"{identity}_explicit")]
            [["question_id", "is_correct", "answer_status"]]
            .rename(columns={"is_correct": "correct_b", "answer_status": "status_b"})
        )
        for variant in [1, 2, 3]:
            implicit = (
                model_detail[model_detail["persona"].eq(f"{identity}_implicit_{variant}")]
                [["question_id", "is_correct", "answer_status"]]
                .rename(columns={"is_correct": "correct_a", "answer_status": "status_a"})
            )
            paired = implicit.merge(explicit, on="question_id", how="inner", validate="one_to_one")
            assert len(paired) == EXPECTED_GSM8K_N
            a = paired["correct_a"].astype(bool).to_numpy()
            b = paired["correct_b"].astype(bool).to_numpy()
            n10 = int(np.sum(a & ~b))
            n01 = int(np.sum(~a & b))
            discordant = n10 + n01
            p_value = float(binomtest(n10, discordant, p=0.5).pvalue) if discordant else 1.0
            p_a, p_b = float(a.mean()), float(b.mean())
            paired_diff = a.astype(int) - b.astype(int)
            se = float(paired_diff.std(ddof=1) / math.sqrt(len(paired_diff)))
            risk_difference_pp = 100 * float(paired_diff.mean())
            paired_rows.append({
                "model": model, "model_label": MODEL_LABELS[model],
                "region": identity, "region_label": REGION_LABELS[identity], "implicit_variant": variant,
                "comparison": f"{REGION_LABELS[identity]} Implicit {variant} - Explicit",
                "n_pairs": len(paired), "correct_implicit": int(a.sum()), "correct_explicit": int(b.sum()),
                "accuracy_implicit_pct": 100 * p_a, "accuracy_explicit_pct": 100 * p_b,
                "risk_difference_pp": risk_difference_pp,
                "risk_difference_ci95_low_pp": risk_difference_pp - 1.959963984540054 * 100 * se,
                "risk_difference_ci95_high_pp": risk_difference_pp + 1.959963984540054 * 100 * se,
                "cohen_h": 2 * math.asin(math.sqrt(p_a)) - 2 * math.asin(math.sqrt(p_b)),
                "discordant_implicit_correct_only": n10,
                "discordant_explicit_correct_only": n01,
                "matched_odds_ratio_haldane": (n10 + 0.5) / (n01 + 0.5),
                "mcnemar_exact_p_value": p_value,
                "commit_rate_implicit_pct": 100 * paired["status_a"].eq("committed").mean(),
                "commit_rate_explicit_pct": 100 * paired["status_b"].eq("committed").mean(),
            })

regional_paired_effects = pd.DataFrame(paired_rows)
regional_paired_effects["q_value_bh_fdr"] = bh_fdr(regional_paired_effects["mcnemar_exact_p_value"])
regional_paired_effects["significant_fdr_0_05"] = regional_paired_effects["q_value_bh_fdr"] < 0.05
regional_paired_effects["abs_cohen_h"] = regional_paired_effects["cohen_h"].abs()
regional_paired_effects["cohen_h_magnitude"] = pd.cut(
    regional_paired_effects["abs_cohen_h"],
    bins=[-np.inf, 0.2, 0.5, 0.8, np.inf],
    labels=["negligible", "small", "medium", "large"], right=False,
)
regional_paired_effects.to_csv(TABLE_DIR / "regional_paired_effects_significance.csv", index=False)

assert len(regional_paired_effects) == 72
assert regional_paired_effects["n_pairs"].eq(EXPECTED_GSM8K_N).all()
assert regional_paired_effects["q_value_bh_fdr"].between(0, 1).all()

display(Markdown("### Regional: hasil berpasangan terkuat setelah BH-FDR"))
display(
    regional_paired_effects.sort_values(
        ["significant_fdr_0_05", "q_value_bh_fdr", "abs_cohen_h"],
        ascending=[False, True, False],
    )[[
        "model_label", "comparison", "accuracy_implicit_pct", "accuracy_explicit_pct",
        "risk_difference_pp", "risk_difference_ci95_low_pp", "risk_difference_ci95_high_pp",
        "cohen_h", "matched_odds_ratio_haldane", "mcnemar_exact_p_value",
        "q_value_bh_fdr", "significant_fdr_0_05",
    ]].head(30).style.format(precision=5)
)

effect_column_order = [
    f"{identity}_implicit_{variant} - {identity}_explicit"
    for identity in REGIONAL_IDENTITIES for variant in [1, 2, 3]
]
regional_paired_effects["short_comparison"] = regional_paired_effects.apply(
    lambda row: f"{row['region']}_implicit_{int(row['implicit_variant'])} - {row['region']}_explicit",
    axis=1,
)
regional_effect_matrix = (
    regional_paired_effects.pivot(index="model_label", columns="short_comparison", values="cohen_h")
    .reindex(index=model_labels_order, columns=effect_column_order)
)
regional_p_matrix = (
    regional_paired_effects.pivot(
        index="model_label", columns="short_comparison", values="mcnemar_exact_p_value"
    ).reindex(index=model_labels_order, columns=effect_column_order)
)
regional_q_matrix = (
    regional_paired_effects.pivot(index="model_label", columns="short_comparison", values="q_value_bh_fdr")
    .reindex(index=model_labels_order, columns=effect_column_order)
)
effect_annotations = regional_effect_matrix.copy().astype(object)
for row_label in regional_effect_matrix.index:
    for column_label in regional_effect_matrix.columns:
        value = regional_effect_matrix.loc[row_label, column_label]
        q_value = regional_q_matrix.loc[row_label, column_label]
        effect_annotations.loc[row_label, column_label] = f"{value:+.02f}{'*' if q_value < 0.05 else ''}"

# Figure 14: effect magnitude and direction; star means q_BH-FDR < .05.
fig, ax = plt.subplots(figsize=(18, 5.8))
effect_limit = max(0.01, np.nanmax(np.abs(regional_effect_matrix.to_numpy())))
sns.heatmap(regional_effect_matrix, annot=effect_annotations, fmt="", center=0,
            vmin=-effect_limit, vmax=effect_limit, cmap=EFFECT_CMAP, linewidths=0.45,
            cbar_kws={"label": "Cohen's h (Implicit - Explicit)"}, ax=ax)
ax.set_title("GSM8K Regional Cultural: Paired Effect Size (* q_BH-FDR < 0,05)")
ax.set_xlabel("Perbandingan persona")
ax.set_ylabel("Model")
ax.tick_params(axis="x", rotation=35)
fig.tight_layout()
save_figure(fig, "14_regional_paired_effect_size_heatmap")
plt.show()

# Figure 15: reference-aligned significance heatmaps. Values below 0.05 are
# dark red; values above 0.05 are clipped to white. The left panel is the raw
# paired exact-McNemar p-value, while the right is the BH-FDR decision value.
def significance_annotations(matrix: pd.DataFrame) -> pd.DataFrame:
    return matrix.map(
        lambda value: ("<.001" if value < 0.001 else f"{value:.3f}")
        + ("*" if value < 0.05 else "")
    )

fig, axes = plt.subplots(1, 2, figsize=(25, 6.2), constrained_layout=True)
for ax, matrix, panel_title, cbar_label in [
    (axes[0], regional_p_matrix, "Exact McNemar p-value", "p-value (0-0,05)"),
    (axes[1], regional_q_matrix, "BH-FDR-corrected q-value", "q-value (0-0,05)"),
]:
    sns.heatmap(
        matrix, annot=significance_annotations(matrix), fmt="",
        cmap=SIGNIFICANCE_CMAP, vmin=0, vmax=0.05, linewidths=0.45,
        cbar_kws={"label": cbar_label}, ax=ax,
    )
    ax.set_title(panel_title)
    ax.set_xlabel("Perbandingan persona")
    ax.set_ylabel("Model")
    ax.tick_params(axis="x", rotation=42)
fig.suptitle(
    "GSM8K Regional Cultural: Significance Value untuk Perbandingan Persona Berpasangan",
    fontsize=15, fontweight="bold",
)
save_figure(fig, "15_regional_mcnemar_pvalue_and_bh_fdr_qvalue_heatmap")
plt.show()
"""
    ),
    code(
        r"""
# Regional token categories use the same equal-weight mean over three implicit
# formulations as the accuracy chart. Underlying persona rows remain in the CSV.
regional_token_rows = []
for model in MODEL_ORDER:
    model_frame = summary[
        summary["dataset"].eq("Regional Cultural") & summary["model"].astype(str).eq(model)
    ].set_index("persona")
    for identity in REGIONAL_IDENTITIES:
        label = REGION_LABELS[identity]
        definitions = {
            f"{label} Explicit": [f"{identity}_explicit"],
            f"{label} Implicit Avg": [f"{identity}_implicit_{variant}" for variant in [1, 2, 3]],
        }
        for category, personas in definitions.items():
            record = {"model": model, "model_label": MODEL_LABELS[model], "category": category}
            for metric in TOKEN_METRICS:
                record[metric] = model_frame.loc[personas, metric].mean()
            regional_token_rows.append(record)
regional_token_categories = pd.DataFrame(regional_token_rows)
regional_token_categories.to_csv(TABLE_DIR / "regional_token_category_summary.csv", index=False)

regional_accuracy_token_columns = [
    column for identity in REGIONAL_IDENTITIES
    for column in [f"{REGION_LABELS[identity]} Explicit", f"{REGION_LABELS[identity]} Implicit Avg"]
]
regional_total_matrix = token_matrix(
    regional_token_categories, "avg_total_tokens", regional_accuracy_token_columns
)
regional_token_delta = pd.DataFrame(index=model_labels_order)
regional_token_delta_raw = []
for identity in REGIONAL_IDENTITIES:
    label = REGION_LABELS[identity]
    explicit = regional_total_matrix[f"{label} Explicit"]
    implicit = regional_total_matrix[f"{label} Implicit Avg"]
    comparison_label = f"{identity}_implicit_avg - {identity}_explicit"
    regional_token_delta[comparison_label] = 100 * (implicit - explicit) / explicit
    for model_label in model_labels_order:
        regional_token_delta_raw.append({
            "model_label": model_label, "region": identity,
            "explicit_avg_total_tokens": explicit.loc[model_label],
            "implicit_avg_total_tokens": implicit.loc[model_label],
            "token_difference": implicit.loc[model_label] - explicit.loc[model_label],
            "token_difference_pct": regional_token_delta.loc[model_label, comparison_label],
        })
pd.DataFrame(regional_token_delta_raw).to_csv(
    TABLE_DIR / "regional_total_token_delta_implicit_vs_explicit.csv", index=False
)

# Figure 16: grouped bars expose sign and within-model regional differences.
fig, ax = plt.subplots(figsize=(15.5, 6.8))
regional_token_delta.plot(kind="bar", width=0.82, color=DELTA_COLORS, ax=ax)
ax.axhline(0, color="black", linewidth=1)
for bar in ax.patches:
    value = bar.get_height()
    ax.annotate(f"{value:+.1f}%", (bar.get_x() + bar.get_width() / 2, value),
                ha="center", va="bottom" if value >= 0 else "top",
                xytext=(0, 4 if value >= 0 else -4), textcoords="offset points", fontsize=7.1)
token_values = regional_token_delta.to_numpy(dtype=float)
token_min = float(np.nanmin(token_values))
token_max = float(np.nanmax(token_values))
token_span = max(1.0, token_max - token_min)
ax.set_ylim(
    0 if token_min >= 0 else token_min - 0.16 * token_span,
    0 if token_max <= 0 else token_max + 0.16 * token_span,
)
ax.set_title("GSM8K Regional Cultural: Total Token Overhead Persona Implicit terhadap Explicit")
ax.set_xlabel("Model")
ax.set_ylabel("Change in average total tokens (%)")
ax.set_xticklabels(model_labels_order, rotation=32, ha="right")
ax.legend(title="Region", bbox_to_anchor=(1.01, 1), loc="upper left")
ax.grid(axis="y", linestyle="--", alpha=0.3)
fig.tight_layout()
save_figure(fig, "16_regional_total_token_delta_implicit_vs_explicit")
plt.show()
"""
    ),
    md(
        r"""
## 6. Visualisasi utama — ditampilkan langsung di notebook

Gambar utama berikut sengaja ditampilkan kembali pada cell tersendiri agar hasilnya langsung
terlihat saat notebook dibuka, bukan hanya tersimpan sebagai PNG/SVG eksternal.
"""
    ),
    code(
        r"""
display(Markdown("### Gender — Perbandingan Akurasi Antar-Persona"))
display(Image(filename=str(FIGURE_DIR / "01_gender_absolute_accuracy_five_categories.png"), width=1400))
"""
    ),
    code(
        r"""
display(Markdown("### Gender — Analisis Delta: Performance Gap dan Gender Bias"))
display(Image(filename=str(FIGURE_DIR / "03_gender_delta_implicit_vs_explicit.png"), width=1400))
"""
    ),
    code(
        r"""
display(Markdown("### Religion — Perbandingan Akurasi Antar-Persona + Rerata Global Persona Neutral"))
display(Image(filename=str(FIGURE_DIR / "05_religion_absolute_accuracy_explicit_implicit.png"), width=1400))
"""
    ),
    code(
        r"""
display(Markdown("### Religion — Analisis Delta: Persona Implicit vs Explicit"))
display(Image(filename=str(FIGURE_DIR / "06_religion_delta_implicit_vs_explicit.png"), width=1400))
"""
    ),
    code(
        r"""
display(Markdown("### Regional Cultural — Perbandingan Akurasi + Rerata Global Persona Neutral"))
display(Image(filename=str(FIGURE_DIR / "12_regional_absolute_accuracy_explicit_implicit.png"), width=1500))
"""
    ),
    code(
        r"""
display(Markdown("### Regional Cultural — Delta Persona Implicit vs Explicit"))
display(Image(filename=str(FIGURE_DIR / "13_regional_delta_implicit_vs_explicit.png"), width=1450))
"""
    ),
    code(
        r"""
display(Markdown("### Regional Cultural — Paired Effect Size (Cohen's h)"))
display(Image(filename=str(FIGURE_DIR / "14_regional_paired_effect_size_heatmap.png"), width=1600))
"""
    ),
    code(
        r"""
display(Markdown("### Regional Cultural — Exact McNemar p-value dan BH-FDR q-value"))
display(Image(filename=str(FIGURE_DIR / "15_regional_mcnemar_pvalue_and_bh_fdr_qvalue_heatmap.png"), width=1800))
"""
    ),
    code(
        r"""
display(Markdown("### Regional Cultural — Total Token Overhead"))
display(Image(filename=str(FIGURE_DIR / "16_regional_total_token_delta_implicit_vs_explicit.png"), width=1450))
"""
    ),
    md(
        r"""
## 7. Pedoman penggunaan di TA/paper

### Tabel utama

- `accuracy_by_persona_six_models.csv`: akurasi, Wilson CI, denominator, dan token per persona.
- `explicit_implicit_descriptive_six_models.csv`: rerata descriptive explicit/implicit per identitas.
- `gender_five_category_and_deltas.csv`: lima kategori utama serta delta terhadap Neutral dan framing.
- `religion_explicit_implicit_and_deltas.csv`: level dan delta Explicit/Implicit per identitas agama.
- `marginal_accuracy_effects_six_models.csv`: risk difference, Cohen's h, dan marginal odds ratio.
- `religion_identity_pair_effects.csv`: effect size enam pasangan identitas agama.
- `token_descriptive_effects_six_models.csv`: selisih dan rasio token antar-persona.
- `token_category_summary_six_models.csv`: prompt/completion/reasoning/total token pada kategori visual utama.
- `gender_token_delta_vs_neutral_pct.csv`: overhead token Gender relatif terhadap Neutral.
- `religion_token_delta_implicit_vs_explicit_pct.csv`: overhead token Religion untuk Implicit relatif terhadap Explicit.
- `regional_accuracy_and_deltas.csv`: level dan delta Regional Explicit/Implicit serta Neutral deskriptif.
- `regional_paired_effects_significance.csv`: 72 perbandingan berpasangan, effect size, exact McNemar p, dan BH-FDR q.
- `regional_token_category_summary.csv`: empat komponen token per kategori Regional.
- `regional_total_token_delta_implicit_vs_explicit.csv`: selisih total token Regional absolut dan persen.
- `model_token_efficiency_six_models.csv`: pooled accuracy dan token per model-dataset.
- `source_manifest.csv`: SHA-256 dan metadata delapan file sumber (dua agregat + enam Regional).

### Batas klaim

1. Gunakan istilah **sensitivitas performa terhadap prompt persona**, bukan langsung menyatakan diskriminasi atau hubungan kausal.
2. Risk difference adalah effect size utama karena paling mudah ditafsirkan dalam poin persentase.
3. Pada Gender/Religion, Cohen's h dan odds ratio bersifat marginal dan tidak memanfaatkan pairing pertanyaan; jangan klaim significance dari dua sumber agregat tersebut.
4. Pada Regional, gunakan `q_value_bh_fdr` sebagai significance value utama dan selalu tampilkan bersama risk difference/Cohen's h. Nilai p mentah bukan keputusan akhir setelah 72 perbandingan.
5. Reasoning token bernilai nol pada CSV Gender, kemungkinan karena provider/parser memasukkannya ke completion atau tidak melaporkannya secara terpisah.
6. Tokenizer dan format reasoning berbeda antar-model. Analisis token lintas model bersifat deskriptif.
7. Rerata tiga implicit persona memberi bobot sama pada setiap formulasi dan bukan tiga sampel independen.
8. Garis Neutral pada Regional berasal dari sumber Gender dan hanya baseline visual deskriptif, bukan pasangan untuk exact McNemar.

### Urutan gambar yang disarankan

1. Level akurasi Gender lima kategori.
2. Delta Gender terhadap Neutral.
3. Delta Gender Implicit − Explicit.
4. Heatmap effect size granular Gender.
5. Level akurasi Religion Explicit vs Implicit.
6. Delta Religion Implicit − Explicit.
7. Effect size antar-identitas Religion.
8–11. Token per kategori dan token delta.
12. Level akurasi Regional Explicit vs Implicit + Global Neutral Avg.
13. Delta Regional Implicit − Explicit.
14. Paired effect size Regional dengan penanda q < 0,05.
15. Heatmap exact McNemar p-value dan BH-FDR q-value Regional.
16. Total token overhead Regional Implicit − Explicit.
"""
    ),
    code(
        r"""
largest_gender = (
    accuracy_effects[
        accuracy_effects["dataset"].eq("Gender") & accuracy_effects["family"].eq("gender_matched")
    ]
    .sort_values("risk_difference_pp", key=lambda values: values.abs(), ascending=False)
    .iloc[0]
)
largest_religion = (
    accuracy_effects[
        accuracy_effects["dataset"].eq("Religion") & accuracy_effects["family"].eq("religion_identity_matched")
    ]
    .sort_values("risk_difference_pp", key=lambda values: values.abs(), ascending=False)
    .iloc[0]
)
largest_regional = regional_paired_effects.sort_values("abs_cohen_h", ascending=False).iloc[0]
regional_significant_count = int(regional_paired_effects["significant_fdr_0_05"].sum())

display(Markdown(
    "### Ringkasan audit otomatis\n\n"
    f"- Input agregat: **246 baris** = 54 Gender + 96 Religion + 96 Regional; "
    f"Regional berasal dari **{len(regional_detail):,} baris tingkat pertanyaan**.\n"
    f"- Effect Gender matched terbesar: **{largest_gender['model_label']}**, "
    f"{largest_gender['label']} = **{largest_gender['risk_difference_pp']:+.2f} pp**, "
    f"Cohen's h = {largest_gender['cohen_h']:+.3f}.\n"
    f"- Effect Religion matched terbesar: **{largest_religion['model_label']}**, "
    f"{largest_religion['label']} = **{largest_religion['risk_difference_pp']:+.2f} pp**, "
    f"Cohen's h = {largest_religion['cohen_h']:+.3f}.\n"
    f"- Effect Regional terbesar: **{largest_regional['model_label']}**, "
    f"{largest_regional['comparison']} = **{largest_regional['risk_difference_pp']:+.2f} pp**, "
    f"Cohen's h = {largest_regional['cohen_h']:+.3f}, "
    f"q_BH-FDR = {largest_regional['q_value_bh_fdr']:.4g}.\n"
    f"- Regional signifikan setelah BH-FDR: **{regional_significant_count}/72** perbandingan. "
    "Effect Gender/Religion tetap deskriptif; Regional memakai pairing question_id."
))

output_index = []
for path in sorted(TABLE_DIR.glob("*.csv")):
    frame = pd.read_csv(path)
    output_index.append({
        "type": "table", "file": path.name, "rows": len(frame),
        "columns": len(frame.columns), "bytes": path.stat().st_size,
    })
for path in sorted(FIGURE_DIR.glob("*")):
    output_index.append({
        "type": "figure", "file": path.name, "rows": np.nan,
        "columns": np.nan, "bytes": path.stat().st_size,
    })
output_index = pd.DataFrame(output_index)
output_index.to_csv(OUTPUT_ROOT / "output_index.csv", index=False)
display(output_index)

# Final gates: no seventh model and no silent schema drift.
assert set(summary["model"].astype(str)) == set(MODEL_ORDER)
assert summary.groupby("dataset", observed=True)["model"].nunique().to_dict() == {
    "Gender": 6, "Regional Cultural": 6, "Religion": 6,
}
assert summary.groupby("dataset").size().to_dict() == {
    "Gender": 54, "Regional Cultural": 96, "Religion": 96,
}
assert accuracy_effects["model"].nunique() == 6
assert token_effects["model"].nunique() == 6
assert not accuracy_effects["model"].isin(set(MODEL_LABELS) - set(MODEL_ORDER)).any()
assert regional_paired_effects["model"].nunique() == 6
assert len(regional_paired_effects) == 72
assert len(list(FIGURE_DIR.glob("*.png"))) == 16
assert len(list(FIGURE_DIR.glob("*.svg"))) == 16
assert (TABLE_DIR / "source_manifest.csv").exists()
print("SEMUA VALIDASI ENAM MODEL UTAMA LULUS")
"""
    ),
]

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, NOTEBOOK_PATH)
print(f"Notebook written: {NOTEBOOK_PATH}")
