from __future__ import annotations
import hashlib, json, os, random, time
from pathlib import Path
import numpy as np
import torch
import yaml


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def seed_everything(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False


def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sha256_file(path, chunk=1<<20):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        while True:
            b=f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str), encoding='utf-8')


def now(): return time.perf_counter()


def ensure_parent(path): Path(path).parent.mkdir(parents=True, exist_ok=True)
