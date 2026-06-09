import math
import numpy as np
from pycatch22 import catch22_all

def split_ts(data: np.ndarray, window_size: int) -> np.ndarray:
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    if data.shape[0] < window_size:
        pad_length = window_size - data.shape[0]
        data = np.pad(data, ((0, pad_length), (0, 0)), mode='constant', constant_values=0)
        return np.expand_dims(data, axis=0)

    modulo = data.shape[0] % window_size
    k = data[modulo:].shape[0] / window_size
    assert math.ceil(k) == k
    data_split = np.array(np.split(data[modulo:], int(k)))
    if modulo != 0:
        first_window = data[:window_size].reshape(1, window_size, -1)
        data_split = np.vstack((first_window, data_split))
    return data_split


def extract_catch22_features(data: np.ndarray) -> np.ndarray:
    if data.ndim == 2:
        data = data[:, 0]
    features = np.array([catch22_all(data)['values']])
    return np.nan_to_num(features)
