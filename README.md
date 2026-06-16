# RouterTS

## Get Started

**Step 1:** Clone this repository and change into its root directory.

**Step 2:** Install the dependencies:

```bash
pip install -r requirements.txt
```

**Step 3:** Install the package:

```bash
pip install -e .
```


## Reproduce the paper results

The window-level meta-feature table for the 619-series training split is provided
(`data/TSB_meta_feature_U_pool20.csv`).

**1. Get the dataset.** Download TSB-AD-U from the
[TSB-AD benchmark](https://thedatumorg.github.io/TSB-AD/) and place one CSV per
series (label in the last column):

```
datasets/TSB-AD-U/<stem>.csv
```

**2. Generate the detector scores** 

```bash
python scripts/generate_scores.py \
    --dataset_dir datasets/TSB-AD-U \
    --file_list data/test_split.csv \
    --score_dir scores
```

Writes `scores/<Detector>/<stem>.npy` for all 20 detectors over the test split.

**3. Train the per-cluster selectors (offline).** 

```bash
python scripts/run_offline.py \
    --meta_csv data/TSB_meta_feature_U_pool20.csv \
    --file_list data/train_split.csv \
    --output_dir out/run
```

Produces (TAGC selects `k`; on this table it is 13, consolidated to 9):

```
out/run/testbed/classifier/classifier_agg_raw_k13.pkl
out/run/weights/agg_raw_k13/SATzilla_Cluster_C{0..8}/{domain}.pkl
```

(`domain` ∈ `ID`, `Environment`, `Facility`, `Finance`, `HumanActivity`,
`Medical`, `Sensor`, `Synthetic`, `Traffic`, `WebService`.)

**4. Run online inference:**

```bash
# In-distribution (ID)
python scripts/run_online.py \
    --classifier_path out/run/testbed/classifier/classifier_agg_raw_k13.pkl \
    --model_dir out/run/weights/agg_raw_k13 \
    --file_list data/test_split.csv \
    --score_dir scores \
    --dataset_dir datasets/TSB-AD-U \
    --output_dir out/predict --domain ID

# Leave-one-domain-out (OOD)
python scripts/run_online.py ... --output_dir out/predict --domain OOD
```

Results land in `out/predict/ID.csv` / `OOD.csv`.

**Detector fusion (`--B`).** By default (`--B 1`) each series uses its single routed detector's raw score.
With `--B b > 1`, RouterTS fuses the top-*b* detectors (ranked by gap-weighted vote probability) via **weighted reciprocal-rank fusion (wRRF)**:

```bash
python scripts/run_online.py ... --domain ID --B 5
```

## More Details

**Configuration.** RouterTS hyperparameters (window size, gap threshold, wRRF
`k0`, Random Forest settings) live in `routerts/config.py`. Candidate detector
hyperparameters use TSB-AD's `Optimal_Uni_algo_HP_dict`. Dataset splits are in
`data/` (`test_split.csv` = 251 test, `train_split.csv` = 619 train,
`TSB_meta_feature_U_pool20.csv` = window-level training meta-features).

**Testbed.** The benchmark splits, evaluation protocols, candidate detector pool
and hyperparameters follow the **TSB-AutoAD** framework (Liu et al.). The
underlying datasets and base anomaly detectors are adopted from the
[TSB-AD benchmark](https://thedatumorg.github.io/TSB-AD/) (NeurIPS 2024).

