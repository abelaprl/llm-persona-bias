import pandas as pd
d8 = pd.read_csv(r'result\reports\meta-llama_llama-3.1-8b-instruct\results.csv')
c8 = (d8['extracted_answer'] == d8['ground_truth_answer']).sum()
t = len(d8)
print(f'8b: {c8}/{t} = {c8/t*100:.2f}%')
d7 = pd.read_csv(r'result\reports\meta-llama_llama-3.3-70b-instruct\results.csv')
c7 = (d7['extracted_answer'] == d7['ground_truth_answer']).sum()
print(f'70b: {c7}/{t} = {c7/t*100:.2f}%')
