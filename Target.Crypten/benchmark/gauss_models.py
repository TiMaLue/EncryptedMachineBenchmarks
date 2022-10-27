
from typing import List, Optional, Callable

import torch

from benchmark import data
from benchmark.gauss_dataset import GaussianClusters
from benchmark.models import Model, binary_classifier_model_output_handler

N_FEATURES = 20

class LogisticRegression(torch.nn.Module):
    def __init__(self, n_features=2):
        super().__init__()
        self.linear = torch.nn.Linear(n_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


class FeedForward(torch.nn.Module):
    def __init__(self, n_features=N_FEATURES):
        super().__init__()
        self.linear1 = torch.nn.Linear(n_features, n_features // 2)
        self.linear2 = torch.nn.Linear(n_features // 2, n_features // 4)
        self.linear3 = torch.nn.Linear(n_features // 4, 1)

    def forward(self, x):
        out = torch.relu(self.linear1(x))
        out = torch.relu(self.linear2(out))
        out = torch.sigmoid(self.linear3(out))
        return out


def std_models(std_models=[]) -> List[Model]:
    if len(std_models) == 0:
        std_models += [
            Model(
                name="logistic regression",
                _file_name="pre_trained_models/logistic_regression.onnx",
                plain=LogisticRegression(),
                data=data.load_dataset("datasets/gauss_clusters_2_1"),
                epochs=10000,
                lr=0.01,
                momentum=0.7,
                loss="BCELoss",
                advanced=False,
                model_output_handler=binary_classifier_model_output_handler
            ),
            Model(
                name="feedforward neural network",
                _file_name="pre_trained_models/feedforward_neural_network.onnx",
                plain=FeedForward(),
                data=data.load_dataset("datasets/gauss_clusters_20_1"),
                epochs=5000,
                lr=0.07,
                momentum=0.8,
                loss="BCELoss",
                advanced=False,
                model_output_handler=binary_classifier_model_output_handler
            )
        ]
    return std_models


def process_model(model, train=False, store_nn=False, load_nn=False, test=False):
    if train:
        model.train()
        model.calc_loss()
    if store_nn:
        model.store_nn()
    if load_nn:
        model.load_nn()
    if test:
        model.calc_loss()


if __name__ == "__main__":
    import sys

    flags = [
        "train", "store_nn",
        "load_nn", "test",
    ]
    flags = [(f, f in sys.argv) for f in flags]
    flags = dict(flags)
    for m in std_models():
        process_model(m, **flags)
