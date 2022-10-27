#!/bin/python3

import json
import sys
import os
from typing import Union, get_origin, Tuple, get_args, Any, List, Type


class TargetParams:
    scenario: str
    plain_model_path: str
    dataset_folder_path: str
    plain_text: bool
    world_size: int
    batch_size: int
    ttp: bool


def start(target_params: TargetParams, output_path: str):
    cmd = "python3 -m benchmark.benchmark_models"
    if target_params.plain_text:
        cmd += " -pt"
    cmd += f" --world-size {target_params.world_size}"
    import tempfile
    temp_output_file = tempfile.mktemp()
    cmd += f" --output-path {temp_output_file}"
    cmd += f" -sc {target_params.scenario}"
    cmd += f" -bs {target_params.batch_size}"
    cmd += f" --plain-model-path {target_params.plain_model_path}"
    cmd += f" --dataset-folder-path {target_params.dataset_folder_path}"
    if target_params.ttp:
        cmd += f" -ttp"

    print("Starting crypten benchmark: " + cmd)
    return_code = os.system(cmd)
    # if return_code != 0:
    #     print("Error within benchmark")
    #     exit(0)
    with open(temp_output_file, "r") as fp:
        measurements = json.load(fp)
    with open(output_path, "w") as fp:
        json.dump(measurements, fp)


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
            print(f"Cannot cast custom value of {attr_name} of class {clz_}. Error: {ex}")
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
    if len(sys.argv) != 3:
        print("Expected 2 inputs got: {}".format(len(sys.argv) - 1))
        exit(1)
    param_path = sys.argv[1]
    with open(param_path, "r") as fp:
        params = json.load(fp)
    if not isinstance(params, dict):
        print("Unexpected params: {}".format(params))
        exit(1)
    print("Running benchmark with params: " + json.dumps(params, indent=2))
    output_path = sys.argv[2]
    target_params = load_from_params(params, TargetParams)
    start(target_params,
          output_path=output_path)
