import json
import random
import pandas as pd
import numpy as np
import os

def detect_question_type(question):
    if not isinstance(question, str):
        return 'other'
    q_lower = question.strip().lower()
    prefixes = ['who', 'what', 'when', 'where', 'why', 'how']
    for prefix in prefixes:
        if q_lower.startswith(prefix):
            return prefix
    return 'other'

def get_answer_length(answers, is_impossible):
    if is_impossible == 1 or not answers:
        return 0
    try:
        first_ans = answers[0].get('text', '').strip()
        if not first_ans:
            return 0
        return len(first_ans.split())
    except (IndexError, AttributeError):
        return 0

def get_answer_length_category(length):
    if length == 0:
        return 'none'
    elif 1 <= length <= 3:
        return 'short'
    elif 4 <= length <= 10:
        return 'medium'
    else:
        return 'long'

def main():
    input_path = r'D:\Abel\TugasAkhir\data\SQUADv2_validation.json'
    output_json = r'D:\Abel\TugasAkhir\data\squad_v2_sampled_1000.json'
    output_csv = r'D:\Abel\TugasAkhir\data\squad_v2_sampled_1000.csv'
    
    if not os.path.exists(input_path):
        print(f'Error: Input file {input_path} not found.')
        return

    print('Loading dataset...')
    with open(input_path, 'r', encoding='utf-8') as f:
        squad_data = json.load(f)
        
    records = []
    for article in squad_data.get('data', []):
        title = article.get('title', '')
        for paragraph in article.get('paragraphs', []):
            context = paragraph.get('context', '')
            for qa in paragraph.get('qas', []):
                q_id = qa.get('id', '')
                question = qa.get('question', '')
                is_impossible = int(qa.get('is_impossible', False))
                answers = qa.get('answers', [])
                
                if not answers or len(answers) == 0:
                    is_impossible = 1
                
                records.append({
                    'id': q_id,
                    'title': title,
                    'context': context,
                    'question': question,
                    'is_impossible': is_impossible,
                    'answers': answers,
                })
                
    df = pd.DataFrame(records)
    print(f'Original dataset size: {len(df)}')
    
    print('Engineering features...')
    df['question_type'] = df['question'].apply(detect_question_type)
    df['answer_length'] = df.apply(lambda row: get_answer_length(row['answers'], row['is_impossible']), axis=1)
    df['answer_length_category'] = df['answer_length'].apply(get_answer_length_category)
    
    df['stratum'] = df['is_impossible'].astype(str) + '_' + df['question_type'] + '_' + df['answer_length_category']
    
    total_d = len(df)
    stratum_counts = df['stratum'].value_counts()
    
    target_total = 1000
    n_s = {}
    
    print('Calculating strata sizes...')
    for s, count in stratum_counts.items():
        n_s[s] = max(1, int(round((count / total_d) * target_total)))
        
    print('Normalizing strata sizes to exactly 1000...')
    while sum(n_s.values()) != target_total:
        current_total = sum(n_s.values())
        if current_total > target_total:
            largest = max(n_s, key=lambda k: n_s[k])
            if n_s[largest] > 1:
                n_s[largest] -= 1
            else:
                break
        elif current_total < target_total:
            largest_orig = max(n_s, key=lambda k: stratum_counts[k])
            n_s[largest_orig] += 1

    print('Sampling dataset...')
    sampled_dfs = []
    for s, count_needed in n_s.items():
        stratum_df = df[df['stratum'] == s]
        sample_size = min(count_needed, len(stratum_df))
        sampled_dfs.append(stratum_df.sample(n=sample_size, random_state=42))
        
    final_df = pd.concat(sampled_dfs).sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    print('\n--- VALIDATION REQUIREMENTS ---')
    print(f'Total rows sampled: {len(final_df)} (Target: {target_total})')
    
    print('\n1. Distribution of is_impossible:')
    print('Sampled:')
    print((final_df['is_impossible'].value_counts(normalize=True) * 100).round(2).astype(str) + '%')
    print('Original:')
    print((df['is_impossible'].value_counts(normalize=True) * 100).round(2).astype(str) + '%')
    
    print('\n2. Distribution of question_type:')
    print('Sampled:')
    print((final_df['question_type'].value_counts(normalize=True) * 100).round(2).astype(str) + '%')
    print('Original:')
    print((df['question_type'].value_counts(normalize=True) * 100).round(2).astype(str) + '%')

    print('\n3. Distribution of answer_length_category:')
    print('Sampled:')
    print((final_df['answer_length_category'].value_counts(normalize=True) * 100).round(2).astype(str) + '%')
    print('Original:')
    print((df['answer_length_category'].value_counts(normalize=True) * 100).round(2).astype(str) + '%')

    print('\n4. Compare original vs sampled distribution:')
    orig_pct = (df['stratum'].value_counts(normalize=True) * 100).to_dict()
    samp_pct = (final_df['stratum'].value_counts(normalize=True) * 100).to_dict()
    
    diff_records = []
    for st in set(orig_pct.keys()).union(samp_pct.keys()):
        o_val = orig_pct.get(st, 0)
        s_val = samp_pct.get(st, 0)
        diff_records.append({'Stratum': st, 'Original %': o_val, 'Sampled %': s_val, 'Diff %': s_val - o_val})
        
    diff_df = pd.DataFrame(diff_records).sort_values(by='Diff %', key=abs, ascending=False)
    print(diff_df.round(3).to_string(index=False))

    final_df_out = final_df.drop(columns=['stratum'])
    
    csv_df = final_df_out.copy()
    csv_df['answers'] = csv_df['answers'].apply(json.dumps)
    csv_df.rename(columns={
        'question_type': 'type',
        'answer_length': 'answer_len',
        'answer_length_category': 'ans_len_cat'
    }, inplace=True)
    csv_df.to_csv(output_csv, index=False)
    
    final_json = []
    for _, row in final_df_out.iterrows():
        final_json.append({
            'id': row['id'],
            'title': row['title'],
            'context': row['context'],
            'question': row['question'],
            'is_impossible': bool(row['is_impossible']),
            'answers': row['answers'],
            'type': row['question_type'],
            'answer_len': row['answer_length'],
            'ans_len_cat': row['answer_length_category']
        })
        
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(final_json, f, indent=4)
        
    print(f'\nSaved JSON to {output_json}')
    print(f'Saved CSV to {output_csv}')

if __name__ == '__main__':
    main()
