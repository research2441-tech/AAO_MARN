from __future__ import annotations
import argparse, json
from pathlib import Path
import cv2, numpy as np, pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from utils import load_config, sha256_file, write_json

IMG_EXT={'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}

def quality(path):
    im=cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None: raise ValueError(f'Unreadable image: {path}')
    h,w=im.shape[:2]; gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY); hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    hist=cv2.calcHist([gray],[0],None,[256],[0,256]).ravel(); p=hist/(hist.sum()+1e-12); p=p[p>0]
    return dict(width=w,height=h,brightness=float(gray.mean()),contrast=float(gray.std()),
                saturation=float(hsv[:,:,1].mean()),entropy=float(-(p*np.log2(p)).sum()),
                blur_laplacian_var=float(cv2.Laplacian(gray,cv2.CV_64F).var()),
                mean_r=float(im[:,:,2].mean()),mean_g=float(im[:,:,1].mean()),mean_b=float(im[:,:,0].mean()))

def build(config_path):
    cfg=load_config(config_path); root=Path(cfg['dataset_root'])
    mapping=pd.read_csv('class_mapping.csv').set_index('original_class')
    rows=[]
    for cls in mapping.index:
        d=root/cls
        if not d.exists(): continue
        meta=mapping.loc[cls]
        for p in sorted(d.rglob('*')):
            if p.suffix.lower() not in IMG_EXT: continue
            q=quality(p)
            rows.append({'path':str(p.resolve()),'original_class':cls,'crop':meta.crop,
                         'binary_class':meta.binary_class,'label':int(meta.label),'sha256':sha256_file(p),**q})
    if not rows: raise RuntimeError('No mapped PlantVillage images found. Check dataset_root and folder names.')
    df=pd.DataFrame(rows)
    dup=df.groupby('sha256').size(); duplicate_hashes=set(dup[dup>1].index)
    # Split unique-content representatives first, then assign exact duplicates to same split.
    reps=df.sort_values('path').drop_duplicates('sha256').copy()
    s=cfg['split']; seed=int(s['seed'])
    train, temp=train_test_split(reps,test_size=1-s['train'],random_state=seed,stratify=reps['original_class'])
    rel=s['test']/(s['val']+s['test'])
    val,test=train_test_split(temp,test_size=rel,random_state=seed,stratify=temp['original_class'])
    split_by_hash={**{h:'train' for h in train.sha256},**{h:'val' for h in val.sha256},**{h:'test' for h in test.sha256}}
    df['split']=df.sha256.map(split_by_hash)
    out=Path(cfg.get('output_dir','.')); out.mkdir(parents=True,exist_ok=True)
    df.to_csv(out/'manifest.csv',index=False)
    sets={k:set(df.loc[df.split==k,'sha256']) for k in ('train','val','test')}
    overlap={f'{a}_{b}':len(sets[a]&sets[b]) for a,b in [('train','val'),('train','test'),('val','test')]}
    audit={'n_images':len(df),'n_unique_hashes':df.sha256.nunique(),'duplicate_hash_groups':len(duplicate_hashes),
           'split_counts':df.split.value_counts().to_dict(),'binary_support':df.groupby(['split','label']).size().to_dict(),
           'hash_overlap':overlap,'seed':seed}
    write_json(out/'split_audit.json',audit)
    if any(overlap.values()): raise RuntimeError(f'Leakage detected: {overlap}')
    return df

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='config.yaml'); a=ap.parse_args(); build(a.config)
