import os

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import torch


class GaussianClusters:

    def __init__(self, n_samples=5000, n_features=20, n_classes=2):
        self.name = f"GuassianCluster(features={n_features}, classes={n_classes})"
        self.n_samples = n_samples
        self.n_features = n_features
        x, x_test, y, y_test = GaussianClusters.generate_data(n_samples, n_features, n_classes)
        self.x, self.y = x, y
        self.x_test, self.y_test = x_test, y_test

    @staticmethod
    def generate_data(n_samples, n_features, n_classes):
        n_redundant = 2
        if n_features < 3:
            n_redundant = 0
        x, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            # by default, 2 features are redundant
            n_informative=n_features - n_redundant,
            n_redundant=n_redundant,
            n_classes=n_classes,
        )
        x = torch.tensor(x).float()
        y = torch.tensor(y).float().unsqueeze(-1)

        return train_test_split(x, y, test_size=0.025)


def generate_datasets():
    from benchmark.data import store_dataset, load_dataset
    ds = GaussianClusters(n_samples=20000)
    dataset_folder_path = "datasets/gauss_clusters_20_1"
    try:
        os.mkdir(dataset_folder_path)
    except FileExistsError:
        pass
    # store_dataset(ds, dataset_folder_path)

    ds = GaussianClusters(n_samples=20000, n_features=2, n_classes=2)
    dataset_folder_path = "datasets/gauss_clusters_2_1"
    try:
        os.mkdir(dataset_folder_path)
    except FileExistsError:
        pass
    # store_dataset(ds, dataset_folder_path)



if __name__ == "__main__":
    generate_datasets()
