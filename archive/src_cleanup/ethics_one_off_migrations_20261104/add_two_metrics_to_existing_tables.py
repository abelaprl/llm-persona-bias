import pandas as pd
from pathlib import Path
root=Path(r"D:\TugasAkhir")
for domain in ('gender','religion','regional'):
 folder=root/'notebooks'/domain/'ethics accuracy analysis'/'tables'
 if domain=='gender':
  paths=[folder/'ethics_condition_summary_source_copy.csv',folder/'accuracy_construct_valid_subsets.csv',folder/'stance_construct_invalid_subsets.csv']
  for p in paths:
   x=pd.read_csv(p)
   x['accuracy_full']=x['n_correct']/x['total']
   x['prop_wrong_committed']=x['pct_answer_wrong']
   x.to_csv(p,index=False)
 else:
  status_path=folder/f'ethics_{domain}_parity_condition_summary.csv'
  source_path=folder/'ethics_accuracy_source_v3.csv'
  status=pd.read_csv(status_path)
  src=pd.read_csv(source_path)[['model','persona','subset','total_responses','n_correct']]
  status=status.drop(columns=['n_correct'],errors='ignore').merge(src,on=['model','persona','subset'],how='left',validate='one_to_one')
  status['total']=status['total_responses']
  status=status.drop(columns=['total_responses'])
  status['accuracy_full']=status['n_correct']/status['total']
  status['prop_wrong_committed']=status['pct_answer_wrong']
  status.to_csv(status_path,index=False)
