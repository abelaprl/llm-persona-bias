import pandas as pd
p = r"e:/Tugas Akhir/TugasAkhir/result/reports/allenai_olmo-3-32b-think/result_allenai_olmo-3-32b-think.csv"
df = pd.read_csv(p, keep_default_na=False)
n = df[df['persona_folder']=='neutral'].copy()
n['ok'] = n['is_correct'].astype(str).str.lower().eq('true')
risky = {'last_number_fallback','currency_final','final_marker'}
mask = (
    n['finish_reason'].astype(str).str.lower().eq('length')
    & n['extraction_reason'].isin(risky)
    & ~n['has_boxed'].astype(str).str.lower().eq('true')
    & ~n['has_hashes'].astype(str).str.lower().eq('true')
)
print('neutral total', len(n))
print('before correct', int(n['ok'].sum()))
print('guard rows', int(mask.sum()))
print('guard currently wrong', int((mask & ~n['ok']).sum()))
print('guard currently correct', int((mask & n['ok']).sum()))
print('after correct', int((n['ok'] & ~mask).sum()))
