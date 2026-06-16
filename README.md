# RouterTS

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

## Reproduce

**1. Dataset.** Download [TSB-AD-U](https://thedatumorg.github.io/TSB-AD/) and place one CSV per series (label in the last column) at `datasets/TSB-AD-U/<stem>.csv`.

**2. Detector scores.**

```bash
python scripts/generate_scores.py \
    --dataset_dir datasets/TSB-AD-U \
    --file_list data/test_split.csv \
    --score_dir scores
```

**3. Train per-cluster selectors (offline).**

```bash
python scripts/run_offline.py \
    --meta_csv data/TSB_meta_feature_U_pool20.csv \
    --file_list data/train_split.csv \
    --output_dir out/run
```

This produces `out/run/testbed/classifier/classifier_agg_raw_k13.pkl` and `out/run/weights/agg_raw_k13/SATzilla_Cluster_C{0..8}/<domain>.pkl` (TAGC selects k=13 on this table).

**4. Online inference.** Results go to `out/predict/ID.csv` (`OOD.csv` for OOD).

```bash
# In-distribution (ID)
python scripts/run_online.py \
    --classifier_path out/run/testbed/classifier/classifier_agg_raw_k13.pkl \
    --model_dir out/run/weights/agg_raw_k13 \
    --file_list data/test_split.csv \
    --score_dir scores \
    --dataset_dir datasets/TSB-AD-U \
    --output_dir out/predict --domain ID
# Leave-one-domain-out: --domain OOD
```

Set `--B b` with `b > 1` to fuse the top-`b` detectors using weighted reciprocal rank fusion. The default `--B 1` uses the score of the single routed detector.

## More Details

- RouterTS hyperparameters live in `routerts/config.py`. Candidate detector hyperparameters use TSB-AD's `Optimal_Uni_algo_HP_dict` (the same set adopted by the TSB-AutoAD framework).
- Splits (`test_split.csv`, `train_split.csv`) and the window-level training meta-features (`TSB_meta_feature_U_pool20.csv`) are in `data/`.
- Benchmark splits, evaluation protocols, and the candidate detector pool follow the **TSB-AutoAD** framework (VLDB 2025). Datasets and base detectors are from the [TSB-AD benchmark](https://thedatumorg.github.io/TSB-AD/) (NeurIPS 2024).
