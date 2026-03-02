import pandas as pd

def why(row):
    if row['error_category'] == 'no_number_found':
        return 'Tidak ada angka final di output.'
    if row['error_category'] == 'truncated_output':
        return 'Output terpotong, tidak ada jawaban akhir.'
    if row['extracted_answer'] == '' or pd.isna(row['extracted_answer']):
        return 'Jawaban kosong.'
    return 'Angka final salah, penjelasan benar.'

results_path = r"result/reports/openai_gpt-oss-20b/results.csv"
output_path = r"result/reports/openai_gpt-oss-20b/wrong_answers_audit.csv"

df = pd.read_csv(results_path)
wrongs = df[df['is_correct'] == False].copy()
wrongs['kenapa_salah'] = wrongs.apply(why, axis=1)
wrongs.to_csv(output_path, index=False)
print(f"Audit selesai. {len(wrongs)} jawaban salah ditulis ke {output_path}")
