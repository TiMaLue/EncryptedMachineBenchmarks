"""
Contains models used for benchmarking
"""
import logging
from dataclasses import dataclass
from typing import List, Optional, Callable

import torch
from torch import Tensor

from benchmark import data


N_FEATURES = 20

def is_image_cls_model(name: str):
    return "image_cls" in name

@dataclass
class Model:
    name: str
    plain: torch.nn.Module
    data: data.Dataset
    epochs: int
    lr: float
    momentum: float
    loss: str
    advanced: bool
    model_output_handler: Callable[[Tensor], Tensor]
    _file_name: Optional[str] = None

    @property
    def file_name(self) -> str:
        if self._file_name is None:
            return f"pre_trained_models/{self.name}.onnx"
        else:
            return self._file_name

    @file_name.setter
    def file_name(self, fn: str):
        self._file_name = fn

    def train(self):
        criterion = getattr(torch.nn, self.loss)()
        optimizer = torch.optim.SGD(self.plain.parameters(), lr=self.lr, momentum=self.momentum)
        print_steps = self.epochs / 10
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            x, y = self.data.x, self.data.y
            output = self.plain(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            # print statistics
            loss_epoch = loss.item()
            if epoch % print_steps == 0:
                print('[%d] loss: %.5f' % (epoch + 1, loss_epoch))

        print(f'Finished Training {self.name}.')

    def store_nn(self):
        input_ = self.data.x_test
        torch.onnx.export(self.plain, input_, self.file_name)
        # torch.onnx.export_to_pretty_string(model.plain, input_, f"pre_traine_models/{model.name}.pretty.onnx")
        print(f'Stored {self.name} with onnx into {self.file_name}')

    def load_nn(self):
        import onnx
        from onnx2pytorch import ConvertModel

        onnx_model = onnx.load(self.file_name)
        pytorch_model = ConvertModel(onnx_model, experimental=True)
        self.plain = pytorch_model
        print(f'Loaded {self.name} from {self.file_name}')

    def calc_loss(self, output=None, y=None):
        criterion = getattr(torch.nn, self.loss)()
        if output is None:
            output = self.plain(self.data.x_test)
        if y is None:
            y = self.data.y_test
        if is_image_cls_model(self.name):
            y = torch.nn.functional.one_hot(y, num_classes=output.shape[1])
            y = y.to(torch.float32)
            output = torch.nn.functional.softplus(output)
            output = torch.nn.functional.normalize(output)

        if "Simple" in self.name:
            output = torch.clamp(output, min=0.0, max=1.0)

        loss_test = criterion(output, y)

        print(f'Model {self.name} test loss {loss_test.item():.5f}')
        return loss_test.item()



def binary_classifier_model_output_handler(out: Tensor):
    return (out > 0.5).float()


def imagenet_classifier_model_output_handler(out: Tensor):
    return out.argmax(1)


def setup_bench_inference_model(scenario: str, plain_model_path: str, dataset_folder_path: str):
    logging.info("Setting up bench model for scenario %s, model path: %s, dataset folder path: %s", scenario,
                 plain_model_path, dataset_folder_path)
    dataset = data.load_dataset(dataset_folder_path)
    handler = binary_classifier_model_output_handler
    if is_image_cls_model(scenario):
        handler = imagenet_classifier_model_output_handler
    m = Model(name=scenario,
              plain=None,  # will be loaded
              data=dataset,
              epochs=0,
              lr=0,
              momentum=0,
              loss="BCELoss",
              advanced=False,
              _file_name=plain_model_path,
              model_output_handler=handler
              )
    m.load_nn()
    m.calc_loss()
    return m

