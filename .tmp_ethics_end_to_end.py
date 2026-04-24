import subprocess
import pandas as pd
from pathlib import Path

base = Path(r"E:/Tugas Akhir/TugasAkhir")
models = [
    "openai_gpt-oss-20b",
    "openai_gpt-oss-120b",
    "qwen_qwen3-next-80b-a3b-thinking",
    "deepseek_deepseek-r1-distill-qwen-32b",
]
empty_tokens = {"none","null","nan","na","n/a","<>","<&gt;"}
py = str(base / ".venv" / "Scripts" / "python.exe")

def is_empty(v):
    if pd.isna(v):
        return True
    s = str(v).strip()
    if s == "":
        return True
    s2 = s.strip('"\'').strip().lower()
    return s2 in empty_tokens

def count_empty(series):
    return int(series.map(is_empty).sum())

def pick_result_path(model):
    model_dir = base / "result" / "ETHICS" / model
    canonical = model_dir / f"result_ethics_{model}.csv"
    rebuilt = model_dir / f"result_ethics_{model}_rebuilt.csv"
    if canonical.exists():
        return canonical
    if rebuilt.exists():
        return rebuilt
    return canonical

def metrics_for_result(path: Path):
    df = pd.read_csv(path)
    total = len(df)
    return {
        "total_rows": int(total),
        "empty_prediction_label": count_empty(df["prediction_label"]) if "prediction_label" in df.columns else total,
        "empty_parsed_answer": count_empty(df["parsed_answer"]) if "parsed_answer" in df.columns else total,
        "empty_explanation": count_empty(df["explanation"]) if "explanation" in df.columns else total,
    }

before = {}
after = {}
rebuild_rc = {}
before_source = {}

for m in models:
    p = pick_result_path(m)
    before_source[m] = str(p)
    before[m] = metrics_for_result(p)

for m in models:
    model_dir = base / "result" / "ETHICS" / m
    result_out = model_dir / f"result_ethics_{m}.csv"
    report_out = model_dir / f"reports_accuracy_{m}.csv"
    cmd = [py, "build_ethics_reports_accuracy.py", "--model_dir", str(model_dir), "--result_out", str(result_out), "--report_out", str(report_out)]
    p = subprocess.run(cmd, cwd=str(base), capture_output=True, text=True)
    rebuild_rc[m] = p.returncode
    if p.returncode != 0:
        print(f"REBUILD_FAILED|{m}|rc={p.returncode}")
        if p.stderr:
            print(p.stderr[-1000:])

for m in models:
    p = base / "result" / "ETHICS" / m / f"result_ethics_{m}.csv"
    after[m] = metrics_for_result(p)

log_path = base / "result" / "ETHICS" / "fix_prediction_from_response_content_log.csv"
fixed_counts = {m: 0 for m in models}
if log_path.exists():
    log_df = pd.read_csv(log_path)
    status_col = next((c for c in log_df.columns if c.lower() == "status"), None)
    model_col = next((c for c in log_df.columns if "model" in c.lower()), None)
    if status_col and model_col:
        fixed_df = log_df[log_df[status_col].astype(str).str.strip().str.lower() == "fixed"]
        grouped = fixed_df.groupby(model_col).size().to_dict()
        for m in models:
            fixed_counts[m] = int(grouped.get(m, 0))

print("SUMMARY_TABLE_START")
print("model|before_source|before_total_rows|after_total_rows|before_empty_prediction_label|after_empty_prediction_label|delta_empty_prediction_label|before_empty_parsed_answer|after_empty_parsed_answer|delta_empty_parsed_answer|before_empty_explanation|after_empty_explanation|delta_empty_explanation|fixed_count")
for m in models:
    b = before[m]
    a = after[m]
    print(f"{m}|{before_source[m]}|{b['total_rows']}|{a['total_rows']}|{b['empty_prediction_label']}|{a['empty_prediction_label']}|{a['empty_prediction_label']-b['empty_prediction_label']}|{b['empty_parsed_answer']}|{a['empty_parsed_answer']}|{a['empty_parsed_answer']-b['empty_parsed_answer']}|{b['empty_explanation']}|{a['empty_explanation']}|{a['empty_explanation']-b['empty_explanation']}|{fixed_counts[m]}")
print("SUMMARY_TABLE_END")

print("REPORT_CHECK_START")
for m in models:
    rp = base / "result" / "ETHICS" / m / f"reports_accuracy_{m}.csv"
    ex = rp.exists()
    rw = len(pd.read_csv(rp)) if ex else -1
    print(f"{m}|exists={ex}|rows={rw}")
print("REPORT_CHECK_END")

print("REBUILD_RC_START")
for m in models:
    print(f"{m}|rc={rebuild_rc[m]}")
print("REBUILD_RC_END")
