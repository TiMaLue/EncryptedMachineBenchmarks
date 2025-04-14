import os

from mpcbenchrunner.utils import Settings


def resolve_model_path(target: str, scenario: str):
    model_path = None
    if target == "tf_encrypted":
        if "image_cls" in scenario:
            model_path = scenario + ".pb"
        if scenario == "SimpleLogisticReg":
            model_path = "SimpleLogisticReg/saved-model.pb"
        if scenario == "SimpleFFNN":
            model_path = "SimpleFFNN/saved-model.pb"
    else:
        if "image_cls" in scenario:
            model_path = scenario + ".onnx"
        if scenario == "SimpleLogisticReg":
            model_path = "SimpleLogisticReg/model.onnx"
        if scenario == "SimpleFFNN":
            model_path = "SimpleFFNN/model.onnx"
        if scenario == "thesis_lenet5":
            model_path = "thesis_lenet5/model"

    if model_path is None:
        raise ValueError("Unrecognized scenario: " + scenario)

    return os.path.join(Settings.benchmark_data_dir, model_path)


def resolve_dataset_path(scenario: str):
    data_path = None
    if "image_cls" in scenario:
        data_path = "image_cls/datasets"
    if scenario == "SimpleLogisticReg":
        data_path = "SimpleLogisticReg/datasets"
    if scenario == "SimpleFFNN":
        data_path = "SimpleFFNN/datasets"
    if scenario == "thesis_lenet5":
        data_path = "thesis_lenet5/datasets"

    if data_path is None:
        raise ValueError("Unrecognized scenario: " + scenario)
    return os.path.join(Settings.benchmark_data_dir, data_path)


def resolve_docker_image(target: str):
    target = target.lower()
    if "crypten" in target:
        return "mpcbenchtarget_crypten"
    if "mpspdz" in target:
        return "mpcbenchtarget_mpspdz"
    if "tf_encrypted" in target:
        return "mpcbenchtarget_tfe"
    raise ValueError("No image for " + target + " is known")
