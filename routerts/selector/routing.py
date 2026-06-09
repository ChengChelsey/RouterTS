import pickle
import warnings
import numpy as np
from sklearn.metrics import pairwise_distances


def predict_cluster(
    catch22_features: np.ndarray,
    classifier_path: str = None,
    clf: dict = None,
) -> int:

    if clf is None:
        if classifier_path is None:
            raise ValueError("Must provide either classifier_path or clf")
        with open(classifier_path, 'rb') as f:
            clf = pickle.load(f)

    method = clf.get('method', 'unknown')
    if method not in ('agglomerative', 'ward'):
        warnings.warn(
            f"predict_cluster() expects agglomerative/ward, got '{method}'.",
            stacklevel=2,
        )

    dataset_feat = catch22_features.mean(axis=0).reshape(1, -1)
    dataset_scaled = clf['scaler'].transform(dataset_feat)
    distances = pairwise_distances(dataset_scaled, clf['centers'])
    return int(distances.argmin())
