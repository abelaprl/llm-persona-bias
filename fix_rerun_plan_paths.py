import os
import pandas as pd

def fix_rerun_plan_paths():
    # Path ke folder JSON dan reports
    json_base = 'result/allenai_olmo-3-32b-think'
    reports_base = 'result/reports/allenai_olmo-3-32b-think'
    df = pd.read_csv('rerun_plan.csv')
    # Cari persona dari reports folder
    persona_dirs = [d for d in os.listdir(reports_base) if os.path.isdir(os.path.join(reports_base, d))]
    persona_set = set(persona_dirs)
    # Mapping question_id ke persona (dari reports csv)
    qid_to_persona = {}
    for persona in persona_dirs:
        csv_path = os.path.join(reports_base, persona, 'results.csv')
        if not os.path.exists(csv_path):
            continue
        try:
            d = pd.read_csv(csv_path, usecols=['question_id'])
        except Exception:
            continue
        for qid in d['question_id'].unique():
            qid_to_persona[int(qid)] = persona
    # Update persona dan json_path di rerun_plan.csv
    new_persona = []
    new_json_path = []
    for idx, row in df.iterrows():
        qid = int(row['question_id'])
        persona = qid_to_persona.get(qid, '')
        new_persona.append(persona)
        if persona:
            json_path = os.path.join(json_base, persona, f'gsm8k_{qid:05d}.json')
        else:
            json_path = row['json_path']
        new_json_path.append(json_path)
    df['persona'] = new_persona
    df['json_path'] = new_json_path
    df.to_csv('rerun_plan.csv', index=False)
    print('rerun_plan.csv sudah diperbaiki dengan path JSON dan persona yang benar.')

if __name__ == '__main__':
    fix_rerun_plan_paths()
