import json
import random
import pandas as pd
import numpy as np

def detect_question_type(question):
    q_lower = str(question).lower().strip()
    prefixes = ['who', 'what', 'when', 'where', 'why', 'how']
    for p in prefixes:
        if q_lower.startswith(p):
            return p
    return 'other'

def get_answer_length(answers, is_impossible):
    if is_impossible or not answers or len(answers) == 0:
        return 0
    first_ans = answers[0].get('text', '')
    if not first_ans:
        return 0
    return len(str(first_ans).split())

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
    
    print("Loading dataset...")
    with open(input_path, 'r', encoding='utf-8') as f:
        squad_data = json.load(f)
        
    records = []
    for article in squad_data.get('data', []):
        title = article.get('title', '')
        for paragraph in article.get('paragraphs', []):
            context = paragraph.get('context', '')
            for qa in paragraph.get('qas', []):
                is_impossible = int(qa.get('is_impossible', False))
                answers = qa.get('answers', [])
                
                # Edge case policy: empty/malformed answers -> treat as unanswerable
                if not answers or len(answers) == 0:
                    is_impossible = 1
                
                records.append({
                    'id': qa['id'],
                    'title': title,
                    'context': context,
                    'question': qa['question'],
                    'is_impossible': is_impossible,
                    'answers': answers,
                })
                
    df = pd.DataFrame(records)
    
    print("Engineering features...")
    df['question_type'] = df['question'].apply(detect_question_type)
    df['answer_length'] = df.apply(lambda row: get_answer_length(row['answers'], row['is_impossible']), axis=1)
    df['answer_length_category'] = df['answer_length'].apply(get_answer_length_category)
    
    # Stratum definition
    df['stratum'] = df['is_impossible'].astype(str) + "_" + df['question_type'] + "_" + df['answer_length_category']
    
    total_d = len(df)
    stratum_counts = df['stratum'].value_counts().to_dict()
    
    target_total = 1000
    n_s = {}
    
    print("Calculating strata sizes...")
    for s, count in stratum_counts.items():
        n_s[s] = max(1, int(round((count / total_d) * target_total)))
        
    # Normalization step
    current_total = sum(n_s.values())
    
    while current_total > target_total:
        largest = max(n_s, key=n_s.get)
        if n_s[largest] > 1:
            n_s[largest] -= 1
            current_total -= 1
        else:
            break
            
    while current_total < target_total:
        largest = max(n_s, key=n_s.get)
        n_s[largest] += 1
        current_total += 1
        
    # Sampling procedure
    print("Sampling...")
    sampled_dfs = []
    for s, count in n_s.items():
        stratum_df = df[df['stratum'] == s]
        sample_size = min(count, len(stratum_df)) 
        sampled_dfs.append(stratum_df.sample(n=sample_size, random_state=42))
        
    final_df = pd.concat(sampled_dfs).sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # Validation Requirements
    print("\n--- VALIDATION REQUIREMENTS ---")
    print(f"Total rows sampled: {len(final_df)} (Target: {target_total})")
    
    print("\n1. Distribution of is_impossible:")
    print("Sampled:")
    print((final_df['is_impossible'].value_counts(normalize=True) * 100).round(2).astype(str) + '%')
    print("\nOriginal:")
    print((df['is_impossible'].value_counts(normalize=True) * 100).round(2).astype(str) + '%')
    
    print("\n2. Distribution of question_type:")
    print("Sampled:")
    print((final_df['question_type'].value_counts(normalize=True) * 100).round(2).astype(str) + '%')
    print("\nOriginal:")
    print((df['question_type'].value_counts(normalize=True) * 100).round(2).astype(str) + '%')

    print("\n3. Distribution of answer_length_category:")
    print("Sampled:")
    print((final_df['answer_length_category'].value_counts(normalize=True) * 100).round(2).astype(str) + '%')
    print("\nOriginal:")
    print((df['answer_length_category'].value_counts(normalize=True) * 100).round(2).astype(str) + '%')

    # Prep for Output
    final_df_out = final_df.drop(columns=['stratum'])
    
    # CSV output
    csv_df = final_df_out.copy()
    csv_df['answers'] = csv_df['answers'].apply(json.dumps)
    csv_df.rename(columns={
        'question_type': 'type',
        'answer_length': 'answer_len',
        'answer_length_category': 'ans_len_cat'
    }, inplace=True)
    csv_df.to_csv(output_csv, index=False)
    
    # JSON output
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
        
    print(f"\nSaved JSON to {output_json}")
    print(f"Saved CSV to {output_csv}")

if __name__ == '__main__':
    main()