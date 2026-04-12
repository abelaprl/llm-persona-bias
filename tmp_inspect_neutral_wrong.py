import pandas as pd
from pathlib import Path
p = Path(r"e:/Tugas Akhir/TugasAkhir/result/reports/openai_gpt-oss-120b/results.csv")
df = pd.read_csv(p, keep_default_na=False)
df['ok'] = df['is_correct'].astype(str).str.lower().eq('true')
sel = df[(df['persona_folder']=='neutral') & (~df['ok'])].head(12)
for _, r in sel.iterrows():
    print('\n===', r['file'], 'reason=', r['extraction_reason'], 'gt=', r['ground_truth_answer'], 'pred=', r['extracted_answer'])
    txt = str(r.get('response_content_full',''))
    print(txt[-1200:])
