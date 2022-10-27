import torchvision
import torch
from torchvision import datasets, transforms, models


# transform=transforms.Compose([
#         transforms.ToTensor(),
# #         transforms.Normalize((0.1307,), (0.3081,))
#         ])
# dataset2 = datasets.ImageNet('../.cache', train=False,
#                        transform=transform, )
#
# dataset2 = datasets.ImageNet('../.cache', train=False,
#                        transform=transform, )

import onnx
from onnx2pytorch import ConvertModel
onnx_model_file_path = "/home/amin/repos/ma-praxis-related/models/vision/classification/mnist/model/"
onnx_model_file_path += "mnist-1.onnx"
onnx_model_file_path = "/home/amin/repos/ma-praxis-related/models/vision/classification/caffenet/model/caffenet-3.onnx"
onnx_model = onnx.load(onnx_model_file_path)
pytorch_model = ConvertModel(onnx_model, experimental=True)

# dataset2 = datasets.ImageNet('../.cache', train=False, )