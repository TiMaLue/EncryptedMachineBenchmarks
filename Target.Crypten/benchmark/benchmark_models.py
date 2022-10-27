import argparse
import json
import logging
from typing import Callable, Tuple

import crypten
import crypten.communicator as comm
import pandas as pd
import torch
from examples import MultiProcessLauncher
from torch import Tensor

from benchmark import models
from benchmark.bench_utils import Runtime, time_it, batch_it, batch_dataset

logging.getLogger().setLevel(0)


class ModelBenchmarks:
    """Benchmarks runtime and accuracy of all defined in models.MODELS
    """

    def __init__(self, scenario: str, do_plaintext: bool, batch_size: int, model: models.Model, device: str = "cpu", ):
        self.scenario = scenario
        self.do_plaintext = do_plaintext
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.model: models.Model = model

    def prepare_nn_model(self):
        if self.do_plaintext:
            m = self.model.plain
            return m
        else:
            with open(self.model.file_name, "rb") as fp:
                model_crypten = crypten.nn.from_onnx(fp)
            model_crypten.encrypt()
            return model_crypten

    def prepare_data(self):
        x = self.model.data.x_test.to(device=self.device)
        y = self.model.data.y_test.to(device=self.device)
        if self.do_plaintext:
            return x, y
        else:
            x = crypten.cryptensor(x)
            return x, y


    def perform_batch_wise_prediction(self, m: Callable[[Tensor], Tensor], x_test: Tensor) \
            -> tuple[list[Tensor], list[Runtime], list[Tensor]]:
        measurements = list()

        batches: list[Tensor] = batch_dataset(x_test, self.batch_size)  # type: ignore

        predict_func = lambda x_: m(x_)
        time_predict = lambda x_: time_it(predict_func, 1, x_)

        batch_time_meas: list[Runtime] = []
        batch_predictions: list[Tensor] = []
        for batch in batches:
            runtime_meas, predictions = time_predict(batch)
            if not self.do_plaintext:
                predictions = predictions.get_plain_text()
            batch_time_meas.append(runtime_meas)
            batch_predictions.append(predictions)
        return batches, batch_time_meas, batch_predictions


    def cal_avg_time_from_measurements(self, batches: list[Tensor], batch_time_meas: list[Runtime]) -> float:
        avg_time_for_pred = 0.0
        for batch, time_meas in zip(batches, batch_time_meas):
            avg_time_for_pred += time_meas.mid / len(batch)
        return avg_time_for_pred / len(batches)

    def calc_acc_from_measurements(self, batch_predictions: list[Tensor], y: Tensor) -> Tuple[float, float]:
        model_outputs: Tensor = torch.cat(batch_predictions)
        predictions: Tensor = self.model.model_output_handler(model_outputs)
        correct = (predictions == y).sum().float()
        accuracy = float((correct / y.shape[0]).cpu().numpy())
        logging.info("Batch predictions has %s many correct from %s many entries. accuracy is: %s",
                     correct,  y.shape[0], accuracy, )
        loss = self.model.calc_loss(model_outputs, y)
        return accuracy, loss


    def run(self, rank: int, ):
        """Runs and stores benchmarks in self.df"""
        # self.__acc_measurements.clear()
        # self.__time_measurements.clear()
        # if self.do_plaintext:
        #     self.measure_inferences_plain(self.model)
        #     logging.info("Performed benchmark for plain, model=%s, rank=%d", self.model.name, rank)
        # else:
        #     self.measure_inferences_enc(self.model)
        #     logging.info("Performed benchmark for encrypted, model=%s, rank=%d", self.model.name, rank)
        m = self.prepare_nn_model()
        x, y = self.prepare_data()
        batches, batch_time_meas, batch_predictions = self.perform_batch_wise_prediction(m, x)
        time_to_predic = self.cal_avg_time_from_measurements(batches, batch_time_meas)
        measured_acc, measured_loss = self.calc_acc_from_measurements(batch_predictions, y)
        return time_to_predic, measured_acc, measured_loss


def get_args():
    """Parses command line arguments"""
    parser = argparse.ArgumentParser(description="Benchmark Functions")
    parser.add_argument("--debug",
                        type=bool,
                        default=False,
                        required=False,
                        help="debug mode")
    parser.add_argument(
        "--plain-model-path",
        type=str,
        required=True,
        help="path to the onnx plaintext model",
    )
    parser.add_argument(
        "--dataset-folder-path",
        type=str,
        required=True,
        help="path to the dataset folder that contains train.csv and test.csv",
    )
    parser.add_argument(
        "--output-path",
        "-o",
        type=str,
        required=False,
        default=None,
        help="path to the output file with the benchmark results",
    )
    parser.add_argument(
        "--world-size",
        "-w",
        type=int,
        required=False,
        default=2,
        help="world size for number of parties",
    )
    parser.add_argument(
        "-bs",
        type=int,
        required=False,
        default=100,
        help="batch size",
    )
    parser.add_argument(
        "-sc",
        required=True,
        help="Scenario to benchmark"
    )
    parser.add_argument(
        "-pt",
        required=False,
        default=False,
        action="store_true",
        help="Activate plaintext"
    )
    parser.add_argument(
        "-ttp",
        required=False,
        default=False,
        action="store_true",
        help="Activate trusted third party"
    )
    args = parser.parse_args()
    return args


def store_results(output_path: str, acc_measurement, measured_loss, time_measurement):
    results = {}
    results["acc"] = acc_measurement
    results["inference_time_s"] = time_measurement
    results["loss"] = measured_loss
    with open(output_path, "w") as fp:
        json.dump(results, fp)


def entry_point(args, rank: int = None):
    """Runs multiparty benchmarks and prints/saves from source 0"""
    if rank is None:
        rank = comm.get().get_rank()

    logging.info("Entrypoint of rank=%d", rank)

    logging.info("Using cpu as device.")

    scenario = args.sc
    do_plaintext = args.pt
    batch_size = args.bs
    plain_model_path = args.plain_model_path
    dataset_folder_path = args.dataset_folder_path
    target_model = models.setup_bench_inference_model(scenario,
                                                      plain_model_path=plain_model_path,
                                                      dataset_folder_path=dataset_folder_path)
    target_benchmark = ModelBenchmarks(scenario=scenario, do_plaintext=do_plaintext,
                                       batch_size=batch_size, model=target_model)
    logging.info("Model to benchmark %s, plaintext: %s", scenario, do_plaintext)

    measured_pred_time, measured_acc, measured_loss = target_benchmark.run(rank, )
    logging.info("Benchmark finished for rank=%d", rank)
    if rank == 0:
        logging.info("Storing results rank=%d", rank)
        pd.set_option("display.precision", 3)
        logging.info("Benchmark results: \n  time to predict: %s\n  accuracy: %s", measured_pred_time, measured_acc)
        if args.output_path:
            store_results(args.output_path, measured_acc, measured_loss, measured_pred_time, )


def main():
    """Runs benchmarks and saves if path is provided"""
    # crypten.init("configs/crypten_config.yaml")
    args = get_args()

    if args.debug:
        entry_point(args, rank=0)
        return
    if args.world_size < 2 and not args.pt:
        raise ValueError(f"World size is {args.world_size}. Expected at least 2.")
    if args.ttp:
        from crypten.config import cfg
        cfg.mpc.provider = "TTP"
        assert crypten.mpc.get_default_provider().NAME == "TTP"
        # crypten.mpc.set_default_provider(crypten.mpc.provider.TrustedThirdParty)

    if args.pt:
        entry_point(args, rank=0)
    else:
        launcher = MultiProcessLauncher(
            args.world_size, entry_point, fn_args=args
        )
        launcher.start()
        launcher.join()
        launcher.terminate()


if __name__ == "__main__":
    main()
