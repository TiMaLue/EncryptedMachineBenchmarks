import json
import logging
import os
import re
import signal
import tempfile
import time
from sys import argv

import cattrs
from docker.models.containers import Container

from mpcbenchrunner import utils
from mpcbenchrunner.utils import Settings, docker_client, CompactJSONEncoder
from mpcbenchrunner.bench import BenchParameters

logger = logging.getLogger()


class Volumes:
    def __init__(self, params: BenchParameters):
        self.__packets_log = tempfile.NamedTemporaryFile(
            delete=Settings.delete_temp_files
        )
        self.__runtime_meas = tempfile.NamedTemporaryFile(
            delete=Settings.delete_temp_files
        )
        self.__runtime_params = tempfile.NamedTemporaryFile(
            delete=Settings.delete_temp_files
        )
        if params.mode_is_priv_inference:
            self.__model_path = params.pretrained_model_path
            self.__dataset_path = params.dataset_folder_path
        else:
            self.__model_path = None
            self.__dataset_path = None

    @property
    def packets_log_file_path(self) -> str:
        return self.__packets_log.name

    @property
    def runtime_measurements_file_path(self) -> str:
        return self.__runtime_meas.name

    @property
    def runtime_params_file_path(self) -> str:
        return self.__runtime_params.name

    @property
    def packets_log_cnt_file_path(self) -> str:
        return "/packets.log"

    @property
    def runtime_measurements_cnt_file_path(self) -> str:
        return "/runtime_measurements.json"

    @property
    def runtime_params_cnt_file_path(self) -> str:
        return "/runtime_params.json"

    @property
    def pretrained_model_cnt_file_path(self) -> str:
        return "/model"

    @property
    def datasets_cnt_dir_path(self) -> str:
        return "/datasets"

    @property
    def mount_options(self) -> dict:
        mounts = {
            self.packets_log_file_path: {
                "bind": self.packets_log_cnt_file_path,
                "mode": "rw",
            },
            self.runtime_measurements_file_path: {
                "bind": self.runtime_measurements_cnt_file_path,
                "mode": "rw",
            },
            self.runtime_params_file_path: {
                "bind": self.runtime_params_cnt_file_path,
                "mode": "ro",
            },
        }
        if self.__model_path is not None:
            mounts[self.__model_path] = {
                "bind": self.pretrained_model_cnt_file_path,
                "mode": "ro",
            }
        if self.__dataset_path is not None:
            mounts[self.__dataset_path] = {
                "bind": self.datasets_cnt_dir_path  # , 'mode': 'ro'
            }
        return mounts

    def __repr__(self):
        return json.dumps(self.mount_options)


def get_tc_env_vars(params: BenchParameters) -> dict:
    vars = dict()
    if params.mtu is not None:
        assert int(params.mtu) >= 1000
        vars["LO_ENABLE_MTU_LIMIT"] = "1"
        vars["LO_MTU"] = params.mtu
    if params.is_tc_enabled:
        if params.tc_rate is not None:
            vars["LO_ENABLE_TC_RATE"] = "1"
            vars["LO_RATE_MBIT"] = params.tc_rate
        if params.tc_delay is not None:
            vars["LO_ENABLE_TC_DEL"] = "1"
            vars["LO_DELAY_MS"] = params.tc_delay
    return vars


def run_container(params) -> (Container, Volumes):
    v = Volumes(params)
    packets_log = tempfile.NamedTemporaryFile(delete=False)
    cont_env_vars = dict()
    cont_env_vars.update(get_tc_env_vars(params))
    # cmd = "watch -n 0.5 ip -s link show dev lo".split(" ")
    logger.debug(
        "running container %s with package log file: %s",
        params.image_name,
        packets_log.name,
    )
    logger.debug(
        "Container volumes: %s",
        json.dumps(v.mount_options, indent=2, cls=CompactJSONEncoder),
    )
    logger.debug(
        "Container env variables: %s",
        json.dumps(cont_env_vars, indent=2, cls=CompactJSONEncoder),
    )
    cont_: Container = docker_client.containers.run(
        params.image_name,
        detach=True,
        volumes=v.mount_options,
        cap_add=["NET_ADMIN"],
        environment=cont_env_vars,
        network_mode="none",
        # cpu_quota=1000000, cpu_period=100000,
    )
    logger.debug(
        f"From image {cont_.image}, started container: {cont_.id}, {cont_.name}"
    )
    return cont_, v


def exec_package_log(cont: Container):
    command = ["/bin/watch_packets.bash", "100"]
    cont.exec_run(command, detach=True)
    time.sleep(0.5)


def read_network_usage_measurements(
    packets_log_file_path: str,
) -> list[(int, list[int], list[int])]:
    with open(packets_log_file_path, "rb") as packets_log_file:
        lines = packets_log_file.readlines()
    entry_line_size = 7
    entries_count = len(lines) / entry_line_size
    logger.debug(
        "Packet logs has %s lines and thus %s entries", len(lines), entries_count
    )
    entries_count = int(entries_count)

    entries = []
    for i in range(0, len(lines), entry_line_size):
        entry_lines = lines[i : i + entry_line_size]
        if len(entry_lines) != entry_line_size:
            logger.warning(
                "Couldn't parse entry. Expected %s lines. Got: %s",
                entry_line_size,
                entry_lines,
            )
            continue
        timestamp = int(entry_lines[0].decode("utf-8").strip())
        rx_numbers = [
            int(m) for m in re.split(r"\D+", entry_lines[4].decode("utf-8"))[1:-1]
        ]
        tx_numbers = [
            int(m) for m in re.split(r"\D+", entry_lines[6].decode("utf-8"))[1:-1]
        ]
        entries.append((timestamp, rx_numbers, tx_numbers))
    if len(entries) > 1:
        logger.debug(
            "Parsed packet log entries in the time span of %s ms",
            entries[-1][0] - entries[0][0],
        )
    return entries


def run_ping(container: Container):
    res = container.exec_run("sleep 2", detach=False, tty=True)
    print(res)


def run_protocol(params: BenchParameters, container: Container, volumes: Volumes):
    runtime_params = dict()
    runtime_params.update(params.params)
    if params.mode_is_priv_inference:
        runtime_params["plain_model_path"] = volumes.pretrained_model_cnt_file_path
        runtime_params["dataset_folder_path"] = volumes.datasets_cnt_dir_path

    with open(volumes.runtime_params_file_path, "w") as fp:
        json.dump(runtime_params, fp)
    logger.info(
        "Running protocol with params: %s",
        json.dumps(runtime_params, indent=2, cls=CompactJSONEncoder),
    )
    docker_exec_cmd = (
        f"./starter.sh"
        f" {volumes.runtime_params_cnt_file_path}"
        f" {volumes.runtime_measurements_cnt_file_path}"
    )
    if params.scheduler_config_path:
        docker_exec_cmd += f" {params.scheduler_config_path}"

    exit_code = os.system(f"docker exec -i {container.id} {docker_exec_cmd}")
    # res = container.exec_run(docker_exec_cmd,
    #                          detach=False)
    # exit_code = res.exit_code
    if exit_code != 0:
        raise RuntimeError("Protocol didn't exist with exit code 0")

    with open(volumes.runtime_measurements_file_path, "r") as fp:
        measurements = json.load(fp)
    return measurements


def compress_packet_stats(packet_stats):
    if len(packet_stats) < Settings.packet_stats_size_ceil:
        return packet_stats
    index = 0
    step_size = len(packet_stats) / Settings.packet_stats_size_ceil
    filtered_packets = []
    while index < len(packet_stats):
        filtered_packets.append(packet_stats[int(index)])
        index += step_size

    logger.debug(
        "Filtered %s packets into %s many packets.",
        len(packet_stats),
        len(filtered_packets),
    )
    return filtered_packets


def raise_on_signal():
    def raise_err(*args):
        raise InterruptedError("MPCBENCHrunner process interrupted")

    signal.signal(signal.SIGINT, raise_err)
    signal.signal(signal.SIGTERM, raise_err)


if __name__ == "__main__":
    from mpcbenchrunner.bench import update_with_packet_stats, BenchParameters

    if len(argv) > 3:
        print("Expected input output files as args.")
        exit(1)
    input_file_path = argv[1]
    output_file_path = argv[2]
    params = BenchParameters.load(input_file_path)
    logger.debug(
        "Starting mpc bench runner with input file: %s and output file: %s",
        input_file_path,
        output_file_path,
    )
    logger.debug("Benchmark params: %s", params)
    cont = None
    try:
        raise_on_signal()
        cont, volumes = run_container(params)
        protocol_measurements = run_protocol(params, cont, volumes)
        exec_package_log(cont)
        packet_stats = read_network_usage_measurements(volumes.packets_log_file_path)
        packet_stats = compress_packet_stats(packet_stats)
        update_with_packet_stats(protocol_measurements, packet_stats)
        with open(output_file_path, "w") as fp:
            json.dump(
                cattrs.unstructure(protocol_measurements),
                fp,
                cls=utils.CompactJSONEncoder,
            )
    finally:
        logger.error("Run interrupted.")
        if cont is not None:
            exit_code = os.system(f"docker stop -t 0 {cont.id}")
            logger.info("Docker stop issued.")
