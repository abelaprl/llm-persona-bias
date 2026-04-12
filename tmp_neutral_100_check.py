import pandas as pd
import re
from pathlib import Path

p = Path(r"e:/Tugas Akhir/TugasAkhir/result/reports/openai_gpt-oss-120b/results.csv")
df = pd.read_csv(p, keep_default_na=False)
df["ok"] = df["is_correct"].astype(str).str.lower().eq("true")
w = df[(df["persona_folder"] == "neutral") & (~df["ok"])].copy()

def nstr(x):
    s = str(x).strip()
    if s == "":
        return ""
    try:
        v = float(s)
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        return ("%.6f" % v).rstrip("0").rstrip(".")
    except Exception:
        return s

def in_resp(val, txt):
    if not val:
        return False
    return re.search(r"(?<!\\d)" + re.escape(val) + r"(?!\\d)", txt) is not None

w["gt"] = w["ground_truth_answer"].map(nstr)
w["pred"] = w["extracted_answer"].map(nstr)
w["resp"] = w["response_content_full"].astype(str)
w["gt_in_response"] = w.apply(lambda r: in_resp(r["gt"], r["resp"]), axis=1)
w["pred_in_response"] = w.apply(lambda r: in_resp(r["pred"], r["resp"]), axis=1)
w["likely_extraction_issue"] = w["gt_in_response"] & (w["gt"] != "") & (w["pred"] != "")

print("neutral_wrong_total =", len(w))
print("likely_extraction_issue_count =", int(w["likely_extraction_issue"].sum()))
print("likely_extraction_issue_ratio =", round(float(w["likely_extraction_issue"].mean()), 4))
print("\nTop extraction reasons among neutral wrong:")
print(w["extraction_reason"].value_counts().head(10).to_string())

show = w[[
    "file", "gt", "pred", "extraction_reason", "error_category",
    "likely_extraction_issue", "gt_in_response", "pred_in_response", "text_tail"
]].head(100).copy()
show["text_tail"] = show["text_tail"].astype(str).str.replace("\n", " ", regex=False).str.slice(0, 95)
out = Path(r"e:/Tugas Akhir/TugasAkhir/result/reports/openai_gpt-oss-120b/neutral_wrong_100_terminal_view.csv")
show.to_csv(out, index=False)
print("\nSaved 100-case view to:", out)
print("\nFIRST 100 CASES:")
for i, row in enumerate(show.itertuples(index=False), start=1):
    print(f"{i:03d} | {row.file} | gt={row.gt} pred={row.pred} | reason={row.extraction_reason} err={row.error_category} | likely_extract={row.likely_extraction_issue} | tail={row.text_tail}")
