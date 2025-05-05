import base64
import gzip
import json
import logging
import os
from typing import Optional

import attrs
import cattrs

from mpcbenchrunner.param_mapping import resolve_dataset_path, resolve_model_path

from mpcbenchrunner import param_mapping
from mpcbenchrunner.utils import Settings

logger = logging.getLogger()


class BenchModes:
    priv_inference = "PrivateInference"


class BenchParameters:
    params: dict[str, str]

    def __init__(self, params: dict[str, str]):
        self.params = params
        assert self.target is not None
        assert self.scenario is not None
        assert self.benchmark_mode is not None

    @property
    def target(self):
        return self.params["target"]

    @property
    def scenario(self):
        return self.params["scenario"]

    @property
    def mtu(self) -> Optional[str]:
        return self.params.get("mtu", None)

    @property
    def is_tc_enabled(self) -> bool:
        return self.tc_delay is not None or self.tc_rate is not None

    @property
    def tc_delay(self) -> Optional[str]:
        delay = self.params.get("tc_delay", "-1")
        if delay == "-1":
            return None
        return delay

    @property
    def tc_rate(self) -> Optional[str]:
        rate = self.params.get("tc_rate", "-1")
        if rate == "-1":
            return None
        return rate

    @property
    def benchmark_mode(self):
        if "benchmark_mode" in self.params:
            return self.params["benchmark_mode"]
        return BenchModes.priv_inference

    @staticmethod
    def load(input_file_path) -> "BenchParameters":
        try:
            with open(input_file_path, "r") as fp:
                input_dict = json.load(fp)
                return BenchParameters(input_dict)
        except Exception as e:
            logger.warning("Couldn't load bench params", exc_info=e)

    @property
    def image_name(self) -> str:
        if self.mode_is_priv_inference:
            return param_mapping.resolve_docker_image(self.target)
        else:
            raise RuntimeError(
                "No image for mode: " + self.benchmark_mode + ". Params: " + str(self)
            )

    @property
    def mode_is_priv_inference(self) -> bool:
        return self.benchmark_mode == "PrivateInference"

    @property
    def pretrained_model_path(self) -> str:
        if self.mode_is_priv_inference:
            return resolve_model_path(self.target, self.scenario)
        else:
            raise RuntimeError("No pretrained models for mode: " + self.benchmark_mode)

    @property
    def dataset_folder_path(self) -> str:
        if self.mode_is_priv_inference:
            return resolve_dataset_path(self.scenario)
        else:
            raise RuntimeError("No dataset for mode: " + self.benchmark_mode)

    @property
    def scheduler_config_path(self) -> str:
        return self.params.get("scheduler_config_path", "")


@attrs.define
class Measurements:
    measurement_start_time_ms: int
    compressed_packet_stats: str
    transmitted_bytes: int
    transmitted_packets: int
    acc: float
    loss: float
    inference_time_s: float


def update_with_packet_stats(
    measurements: dict,
    packet_stats: list,
):
    start_time = packet_stats[0][0]
    transmit_bytes = packet_stats[-1][1][0]
    transmit_packets = packet_stats[-1][1][1]
    packet_stats = [(t - start_time, rx, tx) for t, rx, tx in packet_stats]
    packet_stats_str = json.dumps(packet_stats).encode("utf-8")
    compressed = gzip.compress(packet_stats_str, compresslevel=9)
    compressed_packet_stats_str = base64.b64encode(compressed).decode("utf-8")
    measurements["measurement_start_time_ms"] = start_time
    measurements["compressed_packet_stats"] = compressed_packet_stats_str
    measurements["transmitted_bytes"] = transmit_bytes
    measurements["transmitted_packets"] = transmit_packets
