from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
from utils import load_config,write_json
from evaluate import compute

def verify(cfg,manifest_path='manifest.csv',pred_path=None):
    df=pd.read_csv(manifest_path); checks={}
    checks['required_columns']=all(c in df.columns for c in ['path','original_class','crop','label','sha256','split'])
    sets={s:set(df.loc[df.split==s,'sha256']) for s in ['train','val','test']}
    checks['no_hash_leakage']=not bool((sets['train']&sets['val'])|(sets['train']&sets['test'])|(sets['val']&sets['test']))
    checks['binary_labels_only']=set(df.label.unique()).issubset({0,1}); checks['split_names_valid']=set(df.split.unique()).issubset({'train','val','test'})
    if pred_path and Path(pred_path).exists():
        p=pd.read_csv(pred_path); test=df[df.split=='test']; checks['prediction_count_matches_test']=len(p)==len(test); checks['prediction_paths_match_test']=set(p.path)==set(test.path)
        if {'y_true','probability','prediction'}<=set(p.columns):
            th=0.5
            if Path('authoritative_metrics.json').exists(): th=json.loads(Path('authoritative_metrics.json').read_text()).get('threshold',.5)
            m=compute(p.y_true.values,p.probability.values,th,cfg['evaluation']['calibration_bins']); checks['recomputed_metrics']=m
    checks['all_core_checks_pass']=all(v is True for k,v in checks.items() if isinstance(v,bool))
    write_json('verification_report.json',checks); return checks
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='config.yaml'); ap.add_argument('--manifest',default='manifest.csv'); ap.add_argument('--predictions'); a=ap.parse_args(); print(verify(load_config(a.config),a.manifest,a.predictions))
