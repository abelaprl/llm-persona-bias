#!/usr/bin/env python3
"""
Parse iterative test results and generate detailed CSV/XLSX with questions and answers.
"""
import json
import re
import pandas as pd
from pathlib import Path

def load_gold_data(gold_path):
    """Load gold JSONL into dict keyed by id."""
    gold = {}
    with open(gold_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                gold[item['id']] = item
    return gold

def extract_gsm8k_answer(text):
    """Extract numeric answer from GSM8K gold answer (after ####)."""
    match = re.search(r'####\s*(.+)', text)
    if match:
        return match.group(1).strip()
    return None

def extract_mmlu_answer(answer_idx, gold_item):
    """Convert MMLU numeric answer index to letter."""
    choices = ['A', 'B', 'C', 'D']
    if 0 <= answer_idx < len(choices):
        return choices[answer_idx]
    return None

def parse_results():
    base = Path(__file__).parent.parent
    results_file = base / 'results' / 'test_iterative_responses.json'
    gsm_gold_file = base / 'data' / 'gsm8k_gold.jsonl'
    mmlu_gold_file = base / 'data' / 'mmlu_gold.jsonl'
    
    # Load gold
    gsm_gold = load_gold_data(gsm_gold_file)
    mmlu_gold = load_gold_data(mmlu_gold_file)
    
    # Load results
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    rows = []
    for session in results:
        model_order = session['model_order']
        model_id = session['model_id']
        condition = session['condition']
        dataset = session['dataset']
        
        # Get final call
        final_call = [c for c in session['calls'] if c['type'] == 'final']
        if not final_call:
            continue
        final_call = final_call[0]
        
        model_answer = session.get('extracted_final', '')
        
        # Determine question_id (we need to infer from inputs)
        # For now, we'll use the first question (gsm_0 or mmlu_0) as example
        # In full version, we'd read from input files or session metadata
        if dataset == 'gsm':
            question_id = 'gsm8k_0'
            gold_item = gsm_gold.get(question_id, {})
            question = gold_item.get('question', '')
            gold_answer = extract_gsm8k_answer(gold_item.get('answer', ''))
        else:  # mmlu
            question_id = 'mmlu_0'
            gold_item = mmlu_gold.get(question_id, {})
            question = gold_item.get('question', '')
            gold_answer = extract_mmlu_answer(gold_item.get('answer', -1), gold_item)
        
        # Check correctness
        is_correct = (str(model_answer).strip() == str(gold_answer).strip()) if gold_answer else False
        
        # Warmup info
        warmup_calls = [c for c in session['calls'] if c['type'] == 'warmup']
        num_warmups = len(warmup_calls)
        
        rows.append({
            'model_order': model_order,
            'model_id': model_id,
            'condition': condition,
            'dataset': dataset,
            'question_id': question_id,
            'question': question,
            'gold_answer': gold_answer,
            'model_answer': model_answer,
            'is_correct': is_correct,
            'num_warmups': num_warmups,
            'total_calls': session['total_calls'],
            'avg_latency_sec': round(session['avg_latency_sec'], 3),
            'final_status': final_call.get('status', 0),
        })
    
    df = pd.DataFrame(rows)
    
    # Save CSV
    csv_path = base / 'results' / 'detailed_results.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✅ CSV saved: {csv_path}")
    
    # Save Excel with formatting
    xlsx_path = base / 'results' / 'detailed_results.xlsx'
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Results', index=False)
        
        # Auto-adjust column widths
        worksheet = writer.sheets['Results']
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 80)
            worksheet.column_dimensions[column_letter].width = adjusted_width
    
    print(f"✅ Excel saved: {xlsx_path}")
    
    # Print summary
    print("\n📊 Summary:")
    print(f"Total sessions: {len(df)}")
    if len(df) > 0:
        accuracy = df['is_correct'].mean() * 100
        print(f"Accuracy: {accuracy:.1f}%")
        print(f"\nBy condition:")
        print(df.groupby('condition')['is_correct'].agg(['count', 'mean']))
        print(f"\nBy dataset:")
        print(df.groupby('dataset')['is_correct'].agg(['count', 'mean']))
    
    return df

if __name__ == '__main__':
    df = parse_results()
    print("\n✅ Done! Check results/detailed_results.csv and .xlsx")
    print("\n🔍 Sample rows:")
    print(df[['condition', 'dataset', 'question_id', 'gold_answer', 'model_answer', 'is_correct']].head(10))
