#!/usr/local/bin/python3

import configparser
import shutil
import json
import sys
import os
import time
from typing import Union, get_origin, Tuple, get_args, Any, List, Type, NamedTuple
import numpy as np
import onnx
from onnx import numpy_helper

with open("Protocols/protos.json", "r") as fp:
    proto_list = json.load(fp)


MP_SPDZ_HOME = os.environ["MP_SPDZ_HOME"]


def exec(cmd):
    print("Executing: " + cmd)
    return os.system(cmd)


class TargetParams:
    scenario: str
    protocol: str
    plain_model_path: str
    dataset_folder_path: str
    world_size: int
    batch_size: int


class SchedulerParams:
    config_path: str


class Protocol(NamedTuple):
    id_: str
    name: str
    party_num: int
    is_prime: bool
    is_ring: bool
    is_binary: bool
    is_mal: bool
    is_covert: bool
    is_semi: bool
    script: str
    dishonest_maj: bool


def translate_dataset(source_datasets_path: str, fp):
    count = 0
    features = np.load(os.path.join(source_datasets_path, "test.npy"))
    labels = np.load(os.path.join(source_datasets_path, "test_labels.npy"))
    # for label in labels:
    #     print(int(label[0]), end=" ", file=fp)
    #     count += 1
    # print(file=fp)
    # print(f"DEBUG - Features: {features}")
    # print(f"DEBUG - Labels: {labels}")
    for instance in features:
        for feat in instance:
            print(feat, end=" ", file=fp)
            count += 1
    print(file=fp)
    print(f"Produced {count} many inputs from dataset: {source_datasets_path}")
    return features, labels


def write_model_data(weights_arr, biases_arr, fp):
    input_count = 0
    for w, b in zip(weights_arr, biases_arr):
        w = w.reshape((-1))
        assert len(b.shape) == 1
        for w_ in w:
            print(w_, end=" ", file=fp)
            input_count += 1
        print(file=fp)
        for b_ in b:
            print(b_, end=" ", file=fp)
            input_count += 1
        print(file=fp)
    return input_count

    # model_data_arr = [numpy_helper.to_array(d) for d in model_data]
    # sorted_weight_shapes = [
    #     (10, 20),
    #     (10,),
    #     (5, 10),
    #     (5,),
    #     (1, 5),
    #     (1,),
    # ]
    #
    # model_data_arr_sorted = [None] * len(sorted_weight_shapes)
    # for d in model_data_arr:
    #     correct_post = sorted_weight_shapes.index(d.shape)
    #     model_data_arr_sorted[correct_post] = d
    #
    # # biases_proto = model_data_arr_sorted[:3]
    # # weights_proto = model_data_arr_sorted[3:]
    # weights_arr = model_data_arr_sorted[:3] # [numpy_helper.to_array(w) for w in weights_proto]
    # biases_arr = model_data_arr_sorted[3:]# [numpy_helper.to_array(b) for b in biases_proto]


def read_model_weights_and_biases(model_path):
    model = onnx.load(model_path)
    model_data = list(model.graph.initializer)
    model_data_np = [numpy_helper.to_array(d) for d in model_data]
    weights_arr = []
    biases_arr = []
    for i in range(0, len(model_data_np), 2):
        weights_arr.append(model_data_np[i])
        biases_arr.append(model_data_np[i + 1])
    weights_arr = [w.transpose() for w in weights_arr]
    return weights_arr, biases_arr


def translate_simpleffnn(model_path, fp):
    weights_arr, biases_arr = read_model_weights_and_biases(model_path)
    expected_weight_shapes = [
        ((20, 10), (10,)),
        ((10, 5), (5,)),
        ((5, 1), (1,)),
    ]
    for w, b, expected_shape in zip(weights_arr, biases_arr, expected_weight_shapes):
        assert w.shape == expected_shape[0]
        assert b.shape == expected_shape[1]
    input_count = write_model_data(weights_arr, biases_arr, fp)
    print(
        f"Wrote {input_count} inputs for pretrained model data of {model_path}",
        file=sys.stderr,
    )


def translate_simplelogisticreg(model_path, fp):
    weights_arr, biases_arr = read_model_weights_and_biases(model_path)
    expected_weight_shapes = [
        ((2, 1), (1,)),
    ]
    for w, b, expected_shape in zip(weights_arr, biases_arr, expected_weight_shapes):
        assert w.shape == expected_shape[0]
        assert b.shape == expected_shape[1]
    input_count = write_model_data(weights_arr, biases_arr, fp)
    print(
        f"Wrote {input_count} inputs for pretrained model data of {model_path}",
        file=sys.stderr,
    )


def translate_thesis_lenet5(model_path, fp):
    return


def prepare_input_data(target_params: TargetParams):
    with open("Player-Data/Input-P0-0", "w") as fp:
        features, labels = translate_dataset(target_params.dataset_folder_path, fp)
    model_translation_func = None
    if "simpleffnn" in target_params.scenario.lower():
        model_translation_func = translate_simpleffnn
    elif "simplelogisticreg" in target_params.scenario.lower():
        model_translation_func = translate_simplelogisticreg
    elif "thesis_lenet5" in target_params.scenario.lower():
        model_translation_func = translate_thesis_lenet5
    else:
        raise RuntimeError("Unrecognized scenario: " + target_params.scenario)
    with open("Player-Data/Input-P1-0", "w") as fp:
        model_translation_func(target_params.plain_model_path, fp)
    return features, labels


def resolve_programm_name(target_params: TargetParams):
    if "simpleffnn" in target_params.scenario.lower():
        return "SimpleFFNN"
    if "simplelogisticreg" in target_params.scenario.lower():
        return "SimpleLogisticReg"
    if "thesis_lenet5" in target_params.scenario.lower():
        return "thesis_lenet5"
    else:
        raise ValueError(
            f"Program name could not be resolved for scenario: {target_params.scenario}"
        )


def resolve_mpsdpz_proto(target_params: TargetParams) -> Protocol:
    for proto_ in proto_list:
        if target_params.protocol.lower() in proto_["id_"]:
            return Protocol(**proto_)


def compile_prog(
    target_params: TargetParams,
    dataset_size: int,
    scheduler_config_path: str,
    scheduled_params_path: str,
):
    prog = resolve_programm_name(target_params)
    proto = resolve_mpsdpz_proto(target_params)
    if proto is None:
        raise RuntimeError("No protocol found for " + target_params.protocol)
    os.environ["DATASET_SIZE"] = str(dataset_size)
    bs = target_params.batch_size
    if bs < 0:
        bs = dataset_size
    os.environ["BATCH_SIZE"] = str(bs)
    cmd = f"{MP_SPDZ_HOME}/compile.py"
    if proto.is_binary:
        cmd += " -B 32"
    if proto.is_ring:
        cmd += " -R 64"
    if proto.is_prime:
        cmd += " -F 64"
    cmd += f" {prog}"
    if scheduler_config_path:
        cmd += f" {scheduler_config_path}"
    if scheduled_params_path:
        cmd += f" {scheduled_params_path}"
    print(f"Compile command: {cmd}")
    exit_code = exec(cmd)
    assert exit_code == 0


def run_mpspdz_measure_time(
    target_params: TargetParams,
    dataset_size,
    scheduler_config_path: str,
    scheduled_params_path: str,
) -> float:
    proto = resolve_mpsdpz_proto(target_params)
    prog = resolve_programm_name(target_params)
    if proto.party_num == -3:
        if target_params.world_size < 3:
            raise ValueError(
                "Expected at least 3 parties. Got: " + str(target_params.world_size)
            )
    elif proto.party_num != -1 and target_params.world_size != proto.party_num:
        raise ValueError(
            f"Expected at least {proto.party_num} parties. Got {target_params.world_size}"
        )
    # compile.py includes program args in the compiled program name, thus we append the program args to the program name for execution
    if scheduler_config_path:
        prog = "-".join([prog, scheduler_config_path.replace("/", "_")])
    if scheduled_params_path:
        prog = "-".join([prog, scheduled_params_path.replace("/", "_")])
    cmd = f"{MP_SPDZ_HOME}/Scripts/{proto.script} {prog} -OF predictions"
    os.environ["PLAYERS"] = str(target_params.world_size)
    print(f"Run command: {cmd}")
    start_time = time.time()
    exit_code = exec(cmd)
    assert exit_code == 0
    running_time = time.time() - start_time
    return running_time / dataset_size


def measure_acc(labels, dataset_size) -> float:
    # with open("predictions-P0-0", "r") as fp:
    #     print(fp.readlines())
    with open("predictions-P0-0", "r") as fp:
        # predictions = json.load(fp)
        pred_string = fp.readlines()[1]
    print(f"Read predictions: {pred_string}")
    predictions = json.loads(pred_string)
    print(f"Converted to list: {predictions}")
    predictions = [(1.0 if pred >= 0.5 else 0.0) for pred in predictions]
    correct = sum((1 if pred == test else 0) for pred, test in zip(predictions, labels))
    print(f"Got {correct}/{dataset_size}. Accuracy: {correct / dataset_size}")
    return correct / dataset_size


def read_acc():
    with open("predictions-P0-0", "r") as fp:
        # predictions = json.load(fp)
        # pred_string = fp.readlines()[2]
        acc_plain_string = "-1.0"
        pred_string = "-1.0"
        loss_string = "-1"
        num_correct_string = "-1"
        for line in fp:
            if line.startswith("PlaintextAcc:"):
                acc_plain_string = line.split(":")[1]
            if line.startswith("Accuracy:"):
                pred_string = line.split(":")[1]
            if line.startswith("Loss:"):
                loss_string = line.split(":")[1]
            if line.startswith("Correct:"):
                num_correct_string = line.split(":")[1]
    acc_plain = float(acc_plain_string)
    acc = float(pred_string)
    loss = float(loss_string)
    num_correct = int(num_correct_string)
    return acc, loss, num_correct, acc_plain


def start(
    target_params: TargetParams,
    output_path: str,
    scheduler_config_path: str,
    experiment_id: str,
    scheduled_params_path: str,
):
    config = configparser.ConfigParser()
    config.read(scheduler_config_path)
    with open(scheduled_params_path, "r") as fp:
        scheduled_params = json.load(fp)
    if config["Dataset"]["PrepareData "]:
        print("Preparing inputs.")
        features, labels = prepare_input_data(target_params)
        dataset_size = len(labels)
    elif scheduled_params.get("dataset_size", 0) > 0:
        dataset_size = scheduled_params["dataset_size"]
    else:
        dataset_size = 10
        print("WARNING: No dataset size in scheduled parameters, defaulting to 10!")
    print("Compiling prog.")
    compile_prog(
        target_params, dataset_size, scheduler_config_path, scheduled_params_path
    )
    print("Starting mpspedz benchmark.")
    inference_time = run_mpspdz_measure_time(
        target_params, dataset_size, scheduler_config_path, scheduled_params_path
    )
    print("Calculating accuracy.")
    # acc = measure_acc(labels, dataset_size)
    acc, loss, num_correct, acc_plain = read_acc()
    measurements = {
        "plaintext_acc": acc_plain,
        "acc": acc,
        "inference_time_s": inference_time,
        "loss": loss,
        "num_correct": num_correct,
    }
    print("Finished benchmark. Measurements: " + json.dumps(measurements, indent=4))
    with open(output_path, "w") as fp:
        json.dump(measurements, fp)
    if config["Output"]["FullProgramOutput"]:
        print(f"CWD: {os.getcwd()}")
        shutil.copyfile("/wd/predictions-P0-0", f"/output/full-output-{experiment_id}")


def __env_var_name_of_attr(clazz: Type, attr_name: str, prefix: str) -> str:
    return f"{prefix}{clazz.__name__}_{attr_name}".upper()


def load_from_params(str_params: dict[str, str], clz):
    def cast(clz_, attr_name, attr_str_val) -> Any:
        annotations = getattr(clz_, "__annotations__")
        if attr_name not in annotations:
            return attr_str_val

        declared_type = annotations[attr_name]
        if get_origin(declared_type) is Union:
            union_types: Tuple[Any] = get_args(declared_type)
            if len(union_types) == 2:
                if union_types[0] is None:
                    declared_type = union_types[1]
                elif union_types[1] is None:
                    declared_type = union_types[0]
                else:
                    declared_type = None
            else:
                declared_type = None

        if declared_type is None:
            return attr_str_val
        elif attr_str_val is None:
            raise RuntimeError(f"Expected param {attr_name} was not set")

        if declared_type is bool:
            return attr_str_val == "1"
        try:
            return declared_type(attr_str_val)
        except Exception as ex:
            print(
                f"Cannot cast custom value of {attr_name} of class {clz_}. Error: {ex}"
            )
        return attr_str_val

    def set_params(instance__, clz_):
        attribute_list: List[str] = clz_.__annotations__.keys()
        for attr_name in attribute_list:
            param_val_str = str_params[attr_name]
            param_val = cast(clz_, attr_name, param_val_str)
            setattr(instance__, attr_name, param_val)
        return clz_

    instance_ = clz()
    set_params(instance_, clz)

    return instance_


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Expected 5 inputs got: {}".format(len(sys.argv) - 1))
        exit(1)
    param_path = sys.argv[1]
    with open(param_path, "r") as fp:
        params = json.load(fp)
    if not isinstance(params, dict):
        print("Unexpected params: {}".format(params))
        exit(1)
    print("Running benchmark with params: " + json.dumps(params, indent=2))
    output_path = sys.argv[2]
    scheduler_config_path = sys.argv[3]
    experiment_id = sys.argv[4]
    scheduled_params_path = sys.argv[5]
    target_params = load_from_params(params, TargetParams)
    start(
        target_params,
        output_path=output_path,
        scheduler_config_path=scheduler_config_path,
        experiment_id=experiment_id,
        scheduled_params_path=scheduled_params_path,
    )
