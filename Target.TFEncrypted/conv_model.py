import array
import os
import sys
import onnx
import onnx2keras
import numpy as np
import tensorflow as tf
from onnx import numpy_helper
from tensorflow.python.framework import graph_util

model_file = sys.argv[1]
storage_path = sys.argv[2]
model_onnx = onnx.load(model_file)


def change_node_names(onnx_model):
    for i in range(len(onnx_model.graph.node)):
        for j in range(len(onnx_model.graph.node[i].input)):
            if ":" in onnx_model.graph.node[i].input[j]:
                onnx_model.graph.node[i].input[j] = onnx_model.graph.node[i].input[j].replace(":", "_")

        for j in range(len(onnx_model.graph.node[i].output)):
            if ":" in onnx_model.graph.node[i].output[j]:
                onnx_model.graph.node[i].output[j] = onnx_model.graph.node[i].output[j].replace(":", "_")

    for i in range(len(onnx_model.graph.input)):
        if ":" in onnx_model.graph.input[i].name:
            onnx_model.graph.input[i].name = onnx_model.graph.input[i].name.replace(":", "_")

    for i in range(len(onnx_model.graph.output)):
        if ":" in onnx_model.graph.output[i].name:
            onnx_model.graph.output[i].name = onnx_model.graph.output[i].name.replace(":", "_")

    for i in range(len(onnx_model.graph.initializer)):
        if ":" in onnx_model.graph.initializer[i].name:
            onnx_model.graph.initializer[i].name = onnx_model.graph.initializer[i].name.replace(":", "_")

change_node_names(model_onnx)


input_names = [input_.name for input_ in model_onnx.graph.input]
inputs = [input_ for input_ in model_onnx.graph.input]
input_shape = tuple(d.dim_value for d in inputs[0].type.tensor_type.shape.dim)
random_input = np.random.random(input_shape)
random_input = random_input.astype(dtype="float32")

def fix_missing_tensor_contents(graph_def):
    nodes = graph_def.node
    for x in nodes:
        name = x.name
        dtype = x.attr["dtype"].type
        x_shape = [i.size for i in x.attr["value"].tensor.tensor_shape.dim]
        has_tensor_content = len(x.attr["value"].tensor.tensor_content) > 0
#         print(f"name={name} \t\t dtype={dtype}, x_shape={x_shape} has_tensor_content={has_tensor_content}")
        if dtype == 1 and len(x_shape) > 0 and not has_tensor_content:
            print(f"Node has shape but no tensor_content: {name}")
            float_vals = x.attr["value"].tensor.float_val
            has_floats = len(float_vals) > 0
            if not has_floats:
                raise RuntimeError(f"No tensor_content and not float_val: {name}")
            print(f"Converting float_val to tensor_content: {name}")
            float_vals = np.array(list(float_vals), dtype="float32")
            arr = np.ndarray(shape=x_shape, dtype="float32", buffer=float_vals)
            tensor_content = numpy_helper.from_array(arr)
            x.attr["value"].tensor.tensor_content = tensor_content.raw_data
            arr_1 = array.array("f", x.attr["value"].tensor.tensor_content)
            arr_1 = np.array(arr_1).reshape(x_shape)
            assert arr_1 == arr
#             del x.attr["value"].tensor.float_val

def rename_input_node(frozen_graph, current_name):
    found = False
    for n in frozen_graph.node:
        if n.name == current_name:
            assert not found
            n.name = "ModelInput"
            found = True
        for i in range(len(n.input)):
            if n.input[i] == current_name:
                n.input[i] = "ModelInput"

def rename_output_node(frozen_graph, current_name):
    for n in frozen_graph.node:
        if n.name == current_name:
            n.name = "ModelOut"
            return
    raise ValueError("No output node found with name: " + current_name)

# Taken from https://stackoverflow.com/questions/45466020/how-to-export-keras-h5-to-tensorflow-pb
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph

with tf.compat.v1.Session() as sess:
    model_k = onnx2keras.onnx_to_keras(model_onnx, input_names)
    pred = model_k.__call__(random_input)

    model_input = model_k.input.name.replace(":0", "")
    model_output = model_k.output.name.replace(":0", "")
    # constant_graph = graph_util.convert_variables_to_constants(
    #     sess, sess.graph.as_graph_def(), [model_output]
    # )
    # frozen_graph = graph_util.remove_training_nodes(constant_graph)
    frozen_graph = freeze_session(sess, output_names=[model_output])
    fix_missing_tensor_contents(frozen_graph)
    rename_input_node(frozen_graph, model_input)
    rename_output_node(frozen_graph, model_output)
    from pathlib import Path

    dir_ = Path(storage_path).parent
    os.makedirs(dir_, exist_ok=True)
    with open(storage_path, "wb") as f:
        f.write(frozen_graph.SerializeToString())
