from __future__ import annotations
import argparse, math
import numpy as np,pandas as pd
from scipy.stats import wilcoxon, binomtest

def mcnemar_exact(y,a,b):
    y=np.asarray(y); a=np.asarray(a); b=np.asarray(b); ca=a==y; cb=b==y
    n01=int((~ca&cb).sum()); n10=int((ca&~cb).sum()); n=n01+n10
    p=1.0 if n==0 else binomtest(min(n01,n10),n,0.5,alternative='two-sided').pvalue
    return {'discordant_A_wrong_B_right':n01,'discordant_A_right_B_wrong':n10,'exact_p':float(p)}

def wilcoxon_effect(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float); d=a-b
    if np.allclose(d,0): return {'statistic':0.0,'p':1.0,'matched_rank_biserial':0.0}
    r=wilcoxon(a,b,zero_method='wilcox',alternative='two-sided',method='auto')
    # rank-biserial approximation from signed ranks
    from scipy.stats import rankdata
    ranks=rankdata(np.abs(d[d!=0])); ds=d[d!=0]; wp=ranks[ds>0].sum(); wn=ranks[ds<0].sum(); eff=(wp-wn)/(wp+wn)
    return {'statistic':float(r.statistic),'p':float(r.pvalue),'matched_rank_biserial':float(eff)}

def summarize_runs(path,out='statistical_tests.csv'):
    df=pd.read_csv(path); rows=[]
    if {'variant','seed','val_balanced_accuracy'}<=set(df.columns):
        variants=sorted(df.variant.unique()); base=variants[0]
        pivot=df.pivot(index='seed',columns='variant',values='val_balanced_accuracy')
        for v in variants[1:]:
            x=pivot[base].dropna(); y=pivot[v].dropna(); common=x.index.intersection(y.index); t=wilcoxon_effect(y.loc[common],x.loc[common])
            rows.append({'comparison':f'{v} vs {base}','n_seeds':len(common),'mean_A':float(y.loc[common].mean()),'sd_A':float(y.loc[common].std(ddof=1)),'mean_B':float(x.loc[common].mean()),'sd_B':float(x.loc[common].std(ddof=1)),**t})
    pd.DataFrame(rows).to_csv(out,index=False); return rows
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('csv'); a=ap.parse_args(); summarize_runs(a.csv)
