import os
import pathlib
import sys

import tensorflow as tf
import numpy as np


model_mapping = {
    # "image_cls/alexnet": None,
    # "image_cls/convnext_base": tf.keras.applications.ConvNeXtBase,
    # "image_cls/convnext_large": tf.keras.applications.ConvNeXtLarge,
    # "image_cls/convnext_small": tf.keras.applications.ConvNeXtSmall,
    # "image_cls/convnext_tiny": tf.keras.applications.ConvNeXtTiny,
    # "image_cls/googlenet": None,
    # "image_cls/mnasnet0_5": None,
    # "image_cls/mnasnet1_0": None,
    "image_cls/mobilenet_v2": tf.keras.applications.MobileNetV2,
    # "image_cls/regnet_x_16gf": tf.keras.applications.RegNetX160,
    # "image_cls/regnet_x_1_6gf": tf.keras.applications.RegNetX016,
    # "image_cls/regnet_x_32gf": tf.keras.applications.RegNetX320,
    # "image_cls/regnet_x_3_2gf": tf.keras.applications.RegNetX032,
    # "image_cls/regnet_x_400mf": tf.keras.applications.RegNetX004,
    # "image_cls/regnet_x_800mf": tf.keras.applications.RegNetX008,
    # "image_cls/regnet_x_8gf": tf.keras.applications.RegNetX080,
    # "image_cls/regnet_y_16gf": tf.keras.applications.RegNetY160,
    # "image_cls/regnet_y_1_6gf": tf.keras.applications.RegNetY016,
    # "image_cls/regnet_y_32gf": tf.keras.applications.RegNetY320,
    # "image_cls/regnet_y_3_2gf": tf.keras.applications.RegNetY032,
    # "image_cls/regnet_y_400mf": tf.keras.applications.RegNetY004,
    # "image_cls/regnet_y_800mf": tf.keras.applications.RegNetY008,
    # "image_cls/regnet_y_8gf": tf.keras.applications.RegNetY080,
    "image_cls/resnet101": tf.keras.applications.ResNet101,
    "image_cls/resnet152": tf.keras.applications.ResNet152,
    # "image_cls/resnet18":  None,
    # "image_cls/resnet34": None,
    "image_cls/resnet50": tf.keras.applications.ResNet50,
    # "image_cls/resnext101_32x8d": None,
    # "image_cls/resnext50_32x4d": None,
    # "image_cls/shufflenet_v2_x0_5": None,
    # "image_cls/shufflenet_v2_x1_0": None,
    # "image_cls/squeezenet1_0": None,
    # "image_cls/squeezenet1_1": None,
    # "image_cls/vgg11": None,
    # "image_cls/vgg11_bn": None,
    # "image_cls/vgg13": None,
    # "image_cls/vgg13_bn": None,
    "image_cls/vgg16": tf.keras.applications.VGG16,
    # "image_cls/vgg16_bn": None,
    "image_cls/vgg19": tf.keras.applications.VGG19,
    # "image_cls/vgg19_bn": None,
    # "image_cls/wide_resnet101_2": None,
    # "image_cls/wide_resnet50_2": None
}
def cache_tf_saved_model(model_name, cached_saved_model_path):
    if model_name not in model_mapping or model_mapping[model_name] is None:
        print(f"No pretrained model {model_name} for tf. ")
        exit(1)

    model_keras_builder = model_mapping[model_name]

    tf.keras.backend.set_learning_phase(0)
    tf.keras.backend.set_image_data_format("channels_last")
    tf.keras.backend.set_floatx("float32")

    with tf.compat.v1.Session() as sess:
        model_keras = model_keras_builder(weights="imagenet")
        preds = model_keras.predict(np.random.random((1, 224, 224, 3)))
        model_output = model_keras.output.name.replace(":0", "")
        constant_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
            sess, sess.graph.as_graph_def(), [model_output]
        )
        frozen_graph = tf.compat.v1.graph_util.remove_training_nodes(constant_graph)
        os.makedirs(pathlib.Path(cached_saved_model_path).parent, exist_ok=True)
        with open(cached_saved_model_path, "wb") as f:
            f.write(frozen_graph.SerializeToString())


if __name__ == "__main__":
    destination = sys.argv[1]
    for model_name in model_mapping.keys():
        if model_mapping[model_name] is not None:
            output_file = os.path.join(f"{destination}", f"{model_name}.pb")
            print(f"Downloading and converting model {model_name} into {output_file}")
            cache_tf_saved_model(model_name, output_file)
