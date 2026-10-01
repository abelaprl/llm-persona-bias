import os
import pandas as pd

input_csv = r'e:/Tugas Akhir/TugasAkhir/result/reports/allenai_olmo-3-32b-think/results.csv'
output_base = r'e:/Tugas Akhir/TugasAkhir/result/reports/allenai_olmo-3-32b-think'

def main():
    df = pd.read_csv(input_csv)
    personas = df['persona_folder'].unique()
    summary = {}
    for persona in personas:
        persona_rows = df[df['persona_folder'] == persona]
        persona_folder = os.path.join(output_base, persona)
        os.makedirs(persona_folder, exist_ok=True)
        output_csv = os.path.join(persona_folder, 'results.csv')
        persona_rows.to_csv(output_csv, index=False)
        summary[persona] = len(persona_rows)
    print('Persona summary:')
    for persona, count in summary.items():
        print(f'{persona}: {count} rows, file created at {os.path.join(output_base, persona, "results.csv")}')

if __name__ == '__main__':
    main()
