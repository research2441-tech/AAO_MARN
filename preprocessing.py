from __future__ import annotations
import cv2, numpy as np
from skimage.segmentation import slic
from skimage.color import rgb2lab

def clahe_rgb(img_rgb, clip=2.0, grid=8):
    lab=cv2.cvtColor(img_rgb,cv2.COLOR_RGB2LAB); l,a,b=cv2.split(lab)
    c=cv2.createCLAHE(clipLimit=float(clip),tileGridSize=(int(grid),int(grid))).apply(l)
    return cv2.cvtColor(cv2.merge([c,a,b]),cv2.COLOR_LAB2RGB)

def slic_refine(img_rgb, n_segments=120, compactness=10.0, sigma=1.0):
    labels=slic(img_rgb,n_segments=int(n_segments),compactness=float(compactness),sigma=float(sigma),start_label=0,channel_axis=-1)
    out=np.zeros_like(img_rgb)
    for lab in np.unique(labels):
        m=labels==lab; out[m]=np.median(img_rgb[m],axis=0)
    return out.astype(np.uint8)

def kmeans_segment(img_rgb, k=3, attempts=5, seed=0):
    cv2.setRNGSeed(int(seed)); z=img_rgb.reshape(-1,3).astype(np.float32)
    criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,100,0.2)
    _,lab,cent=cv2.kmeans(z,int(k),None,criteria,int(attempts),cv2.KMEANS_PP_CENTERS)
    return cent[lab.flatten()].reshape(img_rgb.shape).clip(0,255).astype(np.uint8)

def apply_pipeline(img_rgb, cfg, variant='full', seed=0):
    p=cfg['preprocessing']; x=img_rgb.copy()
    if variant in ('clahe','clahe_slic_kmeans','full'):
        x=clahe_rgb(x,p['clahe_clip_limit'],p['clahe_tile_grid'])
    if variant in ('slic','slic_kmeans','clahe_slic_kmeans','full'):
        x=slic_refine(x,p['slic_segments'],p['slic_compactness'],p['slic_sigma'])
    if variant in ('kmeans','slic_kmeans','clahe_slic_kmeans','full'):
        x=kmeans_segment(x,p['kmeans_clusters'],p['kmeans_attempts'],seed)
    return x
