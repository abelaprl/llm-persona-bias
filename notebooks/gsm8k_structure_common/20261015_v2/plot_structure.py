from pathlib import Path
import argparse,re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from structure_cues import IDS
MODELS=['meta-llama_llama-3.1-8b-instruct','meta-llama_llama-3.3-70b-instruct','mistralai_ministral-8b-2512','deepseek_deepseek-r1-distill-qwen-32b','openai_gpt-oss-20b','qwen_qwen3-next-80b-a3b-thinking']
NAMES=['Llama 3.1 8B','Llama 3.3 70B','Ministral 8B','DeepSeek R1 32B','GPT-OSS 20B','Qwen3 Next 80B']
ALIASES={'man':'Laki-laki','woman':'Perempuan','atheist':'Ateis','christian':'Kristen','islam':'Islam','jewish':'Yahudi','east_asian':'Asia Timur','north_american':'Amerika Utara','south_asian':'Asia Selatan','sub_saharan_african':'Afrika Sub-Sahara'}
YLABELS=['Label Step','Nomor berurutan','Bullet berlabel tebal','Subjudul #']
def identity(p):return re.sub(r'_(?:explicit|implicit)_?[123]?$','',p)
def condition(p):return 'N' if p=='neutral' else ('E' if 'explicit' in p else 'I'+p[-1])

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);a=parser.parse_args()
 out=a.run/'figures';out.mkdir(exist_ok=True)
 stats=pd.read_csv(a.run/'paired_statistics.csv');profiles=pd.read_csv(a.run/'profiles.csv')
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
 for domain,ds in stats.groupby('domain',sort=False):
  for comparison in ['I_vs_E','persona_vs_N','between_personas']:
   data=ds[ds.comparison.eq(comparison)]
   pairs=list(dict.fromkeys(zip(data.left,data.right)))
   # Between-persona grids are full notebook diagnostics, not selected main figures.
   fig,axes=plt.subplots(3,2,figsize=(13.4 if len(pairs)>12 else 12,6.0),layout='constrained')
   for ax,model,name in zip(axes.flat,MODELS,NAMES):
    values=np.full((4,len(pairs)),np.nan);sig=np.zeros(values.shape,bool)
    g=data[data.model_id.eq(model)].set_index(['marker_id','left','right'])
    for i,m in enumerate(IDS):
     for j,(left,right) in enumerate(pairs):
      r=g.loc[(m,left,right)];values[i,j]=r.delta;sig[i,j]=r.p_holm<.05
    im=ax.imshow(values,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-50,vcenter=0,vmax=50),aspect='auto')
    for i,j in zip(*np.where(sig)):ax.text(j,i,'*',ha='center',va='center',fontsize=15,color='white' if abs(values[i,j])>27.5 else 'black')
    labels=[]
    for left,right in pairs:
     if comparison=='between_personas':labels.append(ALIASES[identity(left)][:3]+'–'+ALIASES[identity(right)][:3]+' '+condition(left))
     else:labels.append(condition(left))
    ax.set_xticks(range(len(pairs)),labels,rotation=90 if comparison=='between_personas' else 0,fontsize=9)
    ax.set_yticks(range(4),YLABELS,fontsize=10)
    ax.set_title(name,fontsize=12,pad=7)
    if comparison!='between_personas':
     identities=list(dict.fromkeys(identity(left) for left,right in pairs))
     n=3 if comparison=='I_vs_E' else 4
     for k,ident in enumerate(identities):
      ax.text((k*n+(n-1)/2+.5)/len(pairs),-.32,ALIASES[ident].replace(' ','\n'),transform=ax.transAxes,ha='center',va='top',fontsize=9)
      if k:ax.axvline(k*n-.5,color='white',lw=1.4)
    ax.tick_params(length=0)
   cb=fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.65,pad=.02)
   cb.set_label('Delta kemunculan (poin persentase)')
   stem=domain+'_'+comparison
   fig.savefig(out/(stem+'.pdf'),bbox_inches='tight');fig.savefig(out/(stem+'.png'),dpi=150,bbox_inches='tight');plt.close(fig)
  # Full prevalence tables remain available without averaging persona scenarios.
  profiles[profiles.domain.eq(domain)].to_csv(a.run/(domain+'_profiles.csv'),index=False)
 print('Wrote nine consistent comparison figures',flush=True)
if __name__=='__main__':main()
