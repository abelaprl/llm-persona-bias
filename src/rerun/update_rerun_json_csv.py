import os
import json
import pandas as pd

def update_json_and_csv():
    # Cek file rerun_results.csv
    if not os.path.exists('rerun_results.csv'):
        print('rerun_results.csv tidak ditemukan, tidak ada yang diupdate.')
        return
    try:
        rerun_results = pd.read_csv('rerun_results.csv')
    except pd.errors.EmptyDataError:
        print('rerun_results.csv kosong, tidak ada yang diupdate.')
        return
    if rerun_results.empty:
        print('rerun_results.csv tidak ada baris, tidak ada yang diupdate.')
        return
    rerun_plan = pd.read_csv('rerun_plan.csv')
    persona_csv_map = {}
    for _, row in rerun_plan.iterrows():
        persona = row['persona']
        model = row['model']
        persona_csv = f"result/reports/{model}/{persona}/results.csv"
        if os.path.exists(persona_csv):
            persona_csv_map[(model, persona)] = persona_csv
    csv_updates = {}
    for _, row in rerun_results.iterrows():
        qid = int(row['question_id'])
        plan_row = rerun_plan[rerun_plan['question_id'] == qid]
        if plan_row.empty:
            continue
        plan_row = plan_row.iloc[0]
        persona = plan_row['persona']
        model = plan_row['model']
        json_path = plan_row['json_path']
        if os.path.exists(json_path):
            with open(json_path, encoding='utf-8') as f:
                data = json.load(f)
            data['response_content'] = row.get('new_response', data.get('response_content', ''))
            data['new_extracted_answer'] = row.get('new_extracted', None)
            data['new_is_correct'] = row.get('new_is_correct', None)
            data['picked_by_rule'] = row.get('picked_by_rule', '')
            data['picked_span'] = row.get('picked_span', '')
            data['failure_reason'] = row.get('failure_reason', '')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        key = (model, persona)
        if key not in csv_updates:
            csv_updates[key] = {}
        csv_updates[key][qid] = row
    for (model, persona), updates in csv_updates.items():
        csv_path = persona_csv_map.get((model, persona))
        if not csv_path or not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        qid_col = None
        for c in ['question_id', 'id', 'qid']:
            if c in df.columns:
                qid_col = c
                break
        if not qid_col:
            continue
        df.set_index(qid_col, inplace=True)
        for qid, row in updates.items():
            if qid in df.index:
                df.at[qid, 'response_content'] = row.get('new_response', df.at[qid, 'response_content'])
                df.at[qid, 'new_extracted_answer'] = row.get('new_extracted', df.at[qid, 'new_extracted_answer'] if 'new_extracted_answer' in df.columns else None)
                df.at[qid, 'new_is_correct'] = row.get('new_is_correct', df.at[qid, 'new_is_correct'] if 'new_is_correct' in df.columns else None)
                df.at[qid, 'picked_by_rule'] = row.get('picked_by_rule', '')
                df.at[qid, 'picked_span'] = row.get('picked_span', '')
                df.at[qid, 'failure_reason'] = row.get('failure_reason', '')
        df.reset_index(inplace=True)
        df.to_csv(csv_path, index=False)
    print('JSON dan CSV sudah otomatis diupdate jika ada data rerun.')

if __name__ == '__main__':
    update_json_and_csv()
