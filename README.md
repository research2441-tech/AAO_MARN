# AAO-MARN Reproducibility Package

Flat reproducibility package for binary Healthy-vs-Diseased PlantVillage screening using CLAHE, SLIC, K-means, VGG16 spatial tokens, Grasshopper Optimization Algorithm (GOA) channel selection, Multi-Head Attention, bidirectional recurrent spatial modeling, and Artificial Algae Optimization (AAO) hyperparameter search.

## Scientific safeguards

- The locked test set is never used for GOA, AAO, early stopping, calibration fitting, or threshold selection.
- Every test metric is recomputed from `locked_test_predictions.csv`; headline values are never hard-coded.
- SHA-256 duplicate checks are performed across train/validation/test partitions.
- Preprocessing and architectural ablations use matched splits, seeds, and training budgets.
- GOA uses an explicit continuous-to-binary channel mask followed by deterministic cardinality repair (32–256 channels).
- The recurrent layer is explicitly bidirectional and models a deterministic spatial-token order, not disease progression over time.
- Modern baselines are evaluated under the same protocol.
- Runtime reporting separates offline GOA/AAO search from per-image deployment inference.

## Install

```bash
python -m pip install -r requirements.txt
```

## Dataset

Set `dataset_root` in `config.yaml` to a PlantVillage directory whose immediate subdirectories are the original source classes listed in `class_mapping.csv`.

## Reproduce

```bash
python run_all.py --config config.yaml
```

For a lightweight integrity check before expensive training:

```bash
python prepare_manifest.py --config config.yaml
python verify_reproducibility.py --config config.yaml --manifest manifest.csv
```

## Authoritative evaluation rule

`locked_test_predictions.csv` is the sole source for final locked-test metrics. `evaluate.py` creates `authoritative_metrics.json`, confidence intervals, calibration statistics, and the confusion matrix from that file. Any historical/legacy aggregate result must remain clearly separate from this locked-test result.

## Principal outputs

`manifest.csv`, `split_audit.json`, `locked_test_predictions.csv`, `authoritative_metrics.json`, `preprocessing_ablation.csv`, `component_ablation.csv`, `baseline_results.csv`, `optimizer_budget.csv`, `complexity_results.csv`, `statistical_tests.csv`, `goa_history.csv`, `aao_history.csv`, `verification_report.json`.
