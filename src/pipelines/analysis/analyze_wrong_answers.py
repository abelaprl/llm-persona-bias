import pandas as pd
import random

persona_paths = [
    "result/archive/reports_mixed_legacy/qwen_qwen3-next-80b-a3b-thinking/man_explicit_1/results.csv",
    "result/archive/reports_mixed_legacy/qwen_qwen3-next-80b-a3b-thinking/woman_explicit_1/results.csv",
    "result/archive/reports_mixed_legacy/qwen_qwen3-next-80b-a3b-thinking/man_implicit_1/results.csv",
    "result/archive/reports_mixed_legacy/qwen_qwen3-next-80b-a3b-thinking/woman_implicit_1/results.csv"
]

def analyze_wrong_answers():
    for persona_path in persona_paths:
        df = pd.read_csv(persona_path)
        wrong = df[df['is_correct'] != True]
        sample_wrong = wrong.sample(n=min(100, len(wrong)), random_state=42)
        print(f"\n=== {persona_path} ===")
        for idx, row in sample_wrong.iterrows():
            qid = row['question_id']
            gt = row['ground_truth_answer']
            ans = row['extracted_answer']
            reason = ""
            if pd.isna(ans) or str(ans).strip() == "":
                reason = "Jawaban kosong"
            elif str(gt).strip() not in str(ans):
                reason = "Format tidak sesuai atau numeric mismatch"
            else:
                reason = "Lainnya"
            print(f"QID: {qid}\nGT: {gt}\nANS: {ans}\nReason: {reason}\n---")

if __name__ == "__main__":
    analyze_wrong_answers()
