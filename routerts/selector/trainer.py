import os
import re
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

from routerts.config import (
    CANDIDATE_MODEL_SET_20,
    DEFAULT_RF_N_ESTIMATORS,
    DEFAULT_RF_MAX_DEPTH,
)


def train_satzilla_model(
    data_path: str,
    file_list: str,
    save_dir: str,
    domain: str = 'ID',
    cluster_list: str = None,
    cluster: str = None,
    classifier: str = 'random_forest',
    rf_seed: int = 2026,
) -> None:
    # Train a multi-output regressor for one (cluster, domain) combination
    Candidate_Model_Set = CANDIDATE_MODEL_SET_20
    n_candidates = len(Candidate_Model_Set)

    if cluster is not None:
        df_clusters = pd.read_csv(cluster_list)
        cluster_datasets = df_clusters[df_clusters['cluster'] == cluster]['dataset_name'].values.tolist()

        df_domains = pd.read_csv(file_list)
        domain_map = dict(zip(
            df_domains['file_name'].str.replace('.csv', '', regex=False),
            df_domains['domain_name'],
        ))

        if domain == 'ID':
            meta_train_list = cluster_datasets
        else:
            meta_train_list = [ds for ds in cluster_datasets if domain_map.get(ds) != domain]
    elif domain == 'ID':
        meta_train_list = pd.read_csv(file_list)['file_name'].values.tolist()
    else:
        df = pd.read_csv(file_list)
        if domain in df['domain_name'].unique():
            meta_train_list = df[df['domain_name'] != domain]['file_name'].values.tolist()
        else:
            print(f'Domain "{domain}" not found in file list.')
            return

    data = pd.read_csv(data_path, index_col=0)

    train_set_base = set(f.split('.')[0] for f in meta_train_list)
    training_data = data.loc[
        data.index.str.split('_').str[:-1].str.join('_').isin(train_set_base)
    ]

    X_train = training_data.iloc[:, n_candidates:]
    y_train = training_data.iloc[:, :n_candidates]

    X_train = X_train.replace([np.nan, np.inf, -np.inf], 0)

    print(f'  [{cluster or "global"}/{domain}] X_train: {X_train.shape}, y_train: {y_train.shape}')

    if classifier == 'random_forest':
        clf = RandomForestRegressor(
            n_estimators=DEFAULT_RF_N_ESTIMATORS,
            max_depth=DEFAULT_RF_MAX_DEPTH,
            n_jobs=-1,
            verbose=True,
            random_state=rf_seed,
        )
    elif classifier == 'knn':
        clf = KNeighborsRegressor(n_neighbors=5, n_jobs=-1)
    else:
        raise ValueError(f"Unknown classifier: {classifier}")

    clf.fit(X_train, y_train)

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    filename = Path(save_dir) / f"{domain}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(clf, f, pickle.HIGHEST_PROTOCOL)

    print(f'  ✓ Saved: {filename}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train cluster model')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--file_list', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--classifier', type=str, default='random_forest')
    parser.add_argument('--domain', type=str, default='ID')
    parser.add_argument('--cluster_list', type=str, default=None)
    parser.add_argument('--cluster', type=str, default=None)
    parser.add_argument('--rf_seed', type=int, default=2026)

    args = parser.parse_args()
    train_satzilla_model(
        data_path=args.data_path,
        file_list=args.file_list,
        save_dir=args.save_dir,
        domain=args.domain,
        cluster_list=args.cluster_list,
        cluster=args.cluster,
        classifier=args.classifier,
        rf_seed=args.rf_seed,
    )
