import functools
import timeit
from typing import List, Tuple, Any

import attrs
import numpy as np
import torch


@attrs.frozen
class Dataset:
    x: torch.Tensor
    y: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor


@attrs.frozen
class Runtime:
    mid: float
    q1: float
    q3: float

    @classmethod
    def from_times(cls, times):
        return Runtime(mid=np.quantile(times, 0.5),
                       q1=np.quantile(times, 0.25),
                       q3=np.quantile(times, 0.75))

    def runtime_per_sample(self, batch_size: int) -> "Runtime":
        return Runtime(mid=self.mid / batch_size,
                       q1=self.q1 / batch_size,
                       q3=self.q3 / batch_size)


def batch_me(func=None, batch_sizes = (1,)):
    if func is None:
        return functools.partial(batch_me, batch_sizes=batch_sizes)

    @functools.wraps(func)
    def batch_wrapper(*args, **kwargs):
        return_vals = []
        data = kwargs["x"]
        for bs in batch_sizes:
            if bs == -1:
                bs = len(data)
            x = data[:bs]
            kwargs["x"] = x
            return_val = func(*args, **kwargs)
            return_vals.append((return_val, bs))
        return return_vals

    return batch_wrapper


def batch_it(func, data: List, batch_sizes: List[int] = (1, -1)):
    return_vals = []
    for bs in batch_sizes:
        if bs == -1:
            bs = len(data)
        return_val: Runtime = func(data[:bs])
        assert isinstance(return_val, Runtime)
        return_val = return_val.runtime_per_sample(bs)
        return_vals.append(return_val)
    return return_vals


def batch_dataset(data: List, batch_size: int) -> List[List]:
    if batch_size == -1:
        return [data]
    if batch_size < 1:
        raise ValueError("Wrong batch size: " + str(batch_size))
    batches = []
    for i in range(0, len(data), batch_size):
        next_i = min(i + batch_size, len(data))
        batches.append(data[i:next_i])
    return batches


def time_me(func=None, n_loops=2):
    """Decorator returning average runtime in seconds over n_loops

    Args:
        func (function): invoked with given args / kwargs
        n_loops (int): number of times to invoke function for timing

    Returns: tuple of (time in seconds, inner quartile range, function return value).
    """
    if func is None:
        return functools.partial(time_me, n_loops=n_loops)

    # @functools.wraps(func)
    def timing_wrapper(*args, **kwargs):
        times = []
        for _ in range(n_loops):
            start = timeit.default_timer()
            return_val = func(*args, **kwargs)
            times.append(timeit.default_timer() - start)
        runtime = Runtime.from_times(times)
        return runtime
    func.__setattr__("time_it", timing_wrapper)
    return func
    # return timing_wrapper


def time_it(func, n_loops=3, *args, **kwargs) -> Tuple[Runtime, Any]:
    times = []
    return_vals = []
    for _ in range(n_loops):
        start = timeit.default_timer()
        return_val = func(*args, **kwargs)
        times.append(timeit.default_timer() - start)
        return_vals.append(return_val)
    runtime = Runtime.from_times(times)
    return runtime, return_vals[-1]

