from __future__ import annotations
import argparse,pandas as pd
from utils import load_config
from prepare_manifest import build
from complexity import measure
from verify_reproducibility import verify

def main(config):
    cfg=load_config(config); df=build(config)
    # Fast, deterministic integrity and complexity stages are run automatically.
    # Expensive training/ablation/baseline/optimizer stages are separate commands so compute budgets remain explicit.
    measure(cfg)
    verify(cfg,'manifest.csv')
    print('Core reproducibility checks completed.')
    print('Next: python train.py --config config.yaml --manifest manifest.csv --seed 17')
    print('Then evaluate the frozen checkpoint with evaluate.py and run ablation.py/baselines.py as computational budget permits.')
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='config.yaml'); a=ap.parse_args(); main(a.config)
