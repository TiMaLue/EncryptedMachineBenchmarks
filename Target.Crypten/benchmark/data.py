import logging
import os
from typing import NamedTuple

import numpy as np
import torch
from torch import tensor


class Dataset(NamedTuple):
    name: str
    x: torch.Tensor
    y: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor


def store_dataset(ds: Dataset, dataset_folder_path: str):
    # train_set = pd.DataFrame(ds.x.numpy())
    # train_set["y"] = ds.y.numpy()
    #
    # test_set = pd.DataFrame(ds.x_test.numpy())
    # test_set["y"] = ds.y_test.numpy()

    # train_set.to_csv(os.path.join(dataset_folder_path, "train.csv"), index=False)
    # test_set.to_csv(os.path.join(dataset_folder_path, "test.csv"), index=False)

    dataset_file_map = {
        "x": os.path.join(dataset_folder_path, "train.npy"),
        "y": os.path.join(dataset_folder_path, "train_labels.npy"),
        "x_test": os.path.join(dataset_folder_path, "test.npy"),
        "y_test": os.path.join(dataset_folder_path, "test_labels.npy")
    }
    for data_var_name, file_path in dataset_file_map.items():
        data = ds.__getattribute__(data_var_name)
        if data is None:
            logging.warning("Cannot store {} into {} as it is none.", data_var_name, file_path)
            continue
        data_np = data.numpy()
        np.save(file_path, data_np)


def load_dataset(dataset_folder_path: str) -> Dataset:

    # train_df = pd.read_csv(os.path.join(dataset_folder_path, "train.csv"))
    # x = (train_df.iloc[:, :-1]).to_numpy(dtype="float32")
    # y = (train_df.iloc[:, -1:]).to_numpy(dtype="float32")
    # x = torch.tensor(x)
    # y = torch.tensor(y)
    #
    # test_df = pd.read_csv(os.path.join(dataset_folder_path, "test.csv"))
    # x_test = (test_df.iloc[:, :-1]).to_numpy(dtype="float32")
    # y_test = (test_df.iloc[:, -1:]).to_numpy(dtype="float32")
    # x_test = torch.tensor(x_test)
    # y_test = torch.tensor(y_test)

    dataset_file_map = {
        "x": os.path.join(dataset_folder_path, "train.npy"),
        "y": os.path.join(dataset_folder_path, "train_labels.npy"),
        "x_test": os.path.join(dataset_folder_path, "test.npy"),
        "y_test": os.path.join(dataset_folder_path, "test_labels.npy")
    }
    data_map = {}
    for data_var_name, file_path in dataset_file_map.items():
        try:
            data_np = np.load(file_path)
            data_t = tensor(data_np)
            data_map[data_var_name] = data_t
        except Exception:
            data_map[data_var_name] = None

    return Dataset(name=dataset_folder_path, **data_map)


