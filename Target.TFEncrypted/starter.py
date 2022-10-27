#!/bin/env python3

import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Dict, Tuple

import attr
import cattr
import numpy as np
import tensorflow as tf
import tf_encrypted as tfe
from tensorflow.python.platform import gfile
from tensorflow_core.python.framework import importer
from tf_encrypted import RemoteConfig
from tf_encrypted.convert import convert
from tf_encrypted.convert.register import _nodef_to_numpy_array
from tf_encrypted.keras.layers import DepthwiseConv2D

custom_registry = tfe.convert.register.registry()


def relu6(*args):
    print("register relu6 by delegating to relu")
    return custom_registry["Relu"](*args)

custom_registry["Relu6"] = relu6


def depthwise_conv2d(converter, interiors, inputs):
    print("register depthwise_conv2d")
    x_in = converter.outputs[inputs[0]]
    kernel = converter.outputs[inputs[1]]
    k = _nodef_to_numpy_array(kernel)
    conv_op = interiors

    kernel_init = tf.keras.initializers.Constant(k)

    try:
        bias = conv_op.attr["bias"]
        if not bias:
            raise AttributeError()
        b = _nodef_to_numpy_array(bias)
        bias_init = tf.keras.initializers.Constant(b)
        use_bias = True
    except (KeyError, AttributeError):
        use_bias = False
        bias_init = "zeros"

    shape = [i.size for i in kernel.attr["value"].tensor.tensor_shape.dim]

    fmt = conv_op.attr["data_format"].s.decode("ascii")
    fmt = "channels_last" if fmt == "NHWC" else "channels_first"

    strides = int(max(conv_op.attr["strides"].list.i))
    padding = conv_op.attr["padding"].s.decode("ascii")

    layer = DepthwiseConv2D(
        kernel_size=(shape[0], shape[1]),
        strides=strides,
        padding=padding,
        depth_multiplier=1,
        data_format=fmt,
        use_bias=use_bias,
        depthwise_initializer=kernel_init,
        bias_initializer=bias_init,
    )
    return layer(x_in)


custom_registry["DepthwiseConv2dNative"] = depthwise_conv2d


def exec(cmd):
    print("Executing: " + cmd)
    return os.system(cmd)

@attr.define
class TargetParams:
    scenario: str
    protocol: str
    plain_model_path: str
    dataset_folder_path: str
    batch_size: int


def start_servers(target_params: TargetParams) -> Tuple[RemoteConfig, Dict[str, subprocess.Popen]]:
    remote_server_config = {
        "server0": "127.0.0.1:10000",
        "server1": "127.0.0.1:10001",
        "server2": "127.0.0.1:10002",
        "prediction-client": "127.0.0.1:10003",
        "weights-provider": "127.0.0.1:10004",
    }
    remote_config_file = tempfile.NamedTemporaryFile().name
    with open(remote_config_file, "w") as fp:
        json.dump(remote_server_config, fp)
    config = tfe.RemoteConfig.load(remote_config_file)
    tfe.set_config(config)
    try:
        os.mkdir("logs")
    except:
        pass
    server_processes = {}
    for s in remote_server_config.keys():
        cmd = f"{sys.executable} -m tf_encrypted.player {s} --config {remote_config_file} > logs/{s} 2>&1 &"
        p = subprocess.Popen(cmd, shell=True)
        server_processes[s] = p
    return config, server_processes


# def prepare_model(target_params: TargetParams):
#     # Old way of converting the model.
#     # Doesn't work with this version of tf
#     # temp_dir = tempfile.mkdtemp()
#     # exit_code = exec(f"conda run -n conv-env onnx-tf convert -i {target_params['plain_model_path']} -o {temp_dir} ")
#     # assert exit_code == 0
#     # return temp_dir
#     # saved_tf_model = tempfile.NamedTemporaryFile(delete=False).name
#
#     cached_saved_model_path = os.path.join(target_params.cache_dir, target_params.scenario, "saved-model.pb")
#     print("Cached saved model should be on: " + cached_saved_model_path)
#     if os.path.exists(cached_saved_model_path):
#         print("Cached saved model exists. No need to covert.")
#
#         # if "image_cls" in target_params.scenario:
#         #     return cached_saved_model_path.replace("image_cls", "image_cls_3")
#         return cached_saved_model_path
#
#     dir_ = Path(cached_saved_model_path).parent
#     os.makedirs(dir_, exist_ok=True)
#     if "image_cls" in target_params.scenario:
#         print(f"Cached model of image cls not found: {cached_saved_model_path}. Cache all the image models.")
#         raise RuntimeError(f"No cached model: {cached_saved_model_path}")
#     else:
#         exit_code = exec(f"python3 ./conv_model.py {target_params.plain_model_path} {cached_saved_model_path}  ")
#         assert exit_code == 0
#     return cached_saved_model_path

def get_saved_model_path(target_params: TargetParams):
    if not os.path.exists(target_params.plain_model_path):
        raise ValueError("No saved model found: " + target_params.plain_model_path)
    return target_params.plain_model_path

def prepare_input_data(target_params: TargetParams):
    features = np.load(os.path.join(target_params.dataset_folder_path, "test.npy"))
    labels = np.load(os.path.join(target_params.dataset_folder_path, "test_labels.npy"))
    if "image_cls" in target_params.scenario:
        features = np.moveaxis(features, 1, -1)
    return features, labels
    pass


def load_frozen_model(tf_frozen_model_path):
    with gfile.GFile(tf_frozen_model_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def


def batch_dataset(data, batch_size: int):
    if batch_size == -1:
        return [data]
    if batch_size < 1:
        raise ValueError("Wrong batch size: " + str(batch_size))
    batches = []
    for i in range(0, len(data), batch_size):
        next_i = min(i + batch_size, len(data))
        batches.append(data[i:next_i])
    return batches

def decode_pred(target_params: TargetParams, predictions: np.ndarray):
    scenario = target_params.scenario
    if "image_cls" in scenario:
        # pred_top = decode_predictions(predictions, top=1)
        # return np.reshape(pred_top, (-1,1))
        return predictions.argmax(1)
    elif "simple" in scenario.lower():
        if "aby3" in target_params.protocol:
            threshold = 0.5
        else:
            threshold = 0.0
        for i in range(len(predictions)):
            predictions[i][0] = 1.0 if predictions[i][0] >= threshold else 0.0
        return predictions

def resolve_tfe_proto(target_params: TargetParams):
    if "securenn" == target_params.protocol.lower():
        return tfe.protocol.SecureNN
    if "aby3" == target_params.protocol.lower():
        return tfe.protocol.ABY3
    if "pond" == target_params.protocol.lower():
        return tfe.protocol.Pond
    raise RuntimeError("Unknown protocol: " + str(target_params))

def run_tf_image_cls(target_params, graph_def, features, labels):
    input_tensor_name  = graph_def.node[0].name + ":0"
    output_tensor_name = graph_def.node[-1].name + ":0"
    with tf.Graph().as_default():
        importer.import_graph_def(graph_def, name="")
        with tf.Session() as sess:
            input_node = sess.graph.get_tensor_by_name(input_tensor_name)
            output_node = sess.graph.get_tensor_by_name(output_tensor_name)
            raw_predictions = sess.run(output_node, feed_dict={input_node: features})
    predictions = raw_predictions.argmax(1)
    correct_pred = sum(predictions == labels)
    print("Plaintext acc: " + str( correct_pred / len(features)))
    return raw_predictions

def run_tfe(target_params, graph_def, features, labels, remote_config):
    batch_size = int(target_params.batch_size)
    # if batch_size == -1:
    #     batch_size = len(features)
    batches = batch_dataset(features, batch_size)
    proto_builder = resolve_tfe_proto(target_params)
    conversion_time = 0.0
    pred_time = 0.0
    with proto_builder() as prot:

        pred_list = []
        for batch in batches:
            tf.reset_default_graph()
            start_time = time.time()
            def provide_input():
                return tf.constant(batch)
            c = convert.Converter(
                custom_registry,
                config=remote_config,
                protocol=prot,
                model_provider=remote_config.get_player("weights-provider"),
            )
            x = c.convert(graph_def, remote_config.get_player("prediction-client"), provide_input)
            conversion_time += time.time() - start_time
            start_time = time.time()
            with tfe.Session(config=remote_config) as sess:
                sess.run(tfe.global_variables_initializer())
                preds = sess.run(x.reveal())
                pred_list.append(preds)
            pred_time += time.time() - start_time


    end_time = time.time() -start_time
    pred_time_sec = end_time/len(features)
    raw_predictions = np.concatenate(pred_list, axis=0)
    predictions = decode_pred(target_params, raw_predictions)
    correct_pred_arr = predictions.reshape(-1, 1) == labels.reshape(-1, 1)
    correct_pred = sum(correct_pred_arr)
    acc = correct_pred / len(features)
    acc = acc.item()
    return conversion_time, pred_time_sec, acc, raw_predictions

def measure_precision_loss(plaintext_pred, enc_pred):
    if plaintext_pred is None or enc_pred is None:
        return None
    assert plaintext_pred.shape == enc_pred.shape
    diff = plaintext_pred - enc_pred
    abs_ = np.absolute(diff)
    return np.average(abs_, axis=(0,1))


def start(target_params: TargetParams, remote_config, output_path: str):
    print("Preparing model and inputs")
    tf_frozen_model_path = get_saved_model_path(target_params)
    features, labels = prepare_input_data(target_params)
    graph_def = load_frozen_model(tf_frozen_model_path)

    print("Start benchmark")
    plaintext_pred = None
    if "image_cls" in target_params.scenario:
        plaintext_pred = run_tf_image_cls(target_params, graph_def, features, labels, )
    conversion_time, inference_time, acc, enc_pred = run_tfe(target_params, graph_def, features, labels, remote_config)

    avg_precision_loss = measure_precision_loss(plaintext_pred, enc_pred)

    measurements = {
        "acc": acc,
        "model_conversion_time": conversion_time,
        "inference_time_s": inference_time,
        "precision_loss": avg_precision_loss
    }
    print("Finished benchmark. Measurements: " + json.dumps(measurements, indent=4))
    with open(output_path, "w") as fp:
        json.dump(measurements, fp, indent=2)


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
    tp = cattr.structure(params, TargetParams)
    print("Running benchmark with params: " + json.dumps(params, indent=2))
    print("Running benchmark with tp: " + str(tp))
    output_path = sys.argv[2]

    remote_config, servers = start_servers(tp)
    try:
        start(tp, remote_config,
              output_path=output_path)
    finally:
        print("Exiting servers")
        for p in servers.values():
            p.terminate()
