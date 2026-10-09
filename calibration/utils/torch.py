import abc
import os
import traceback
from contextlib import contextmanager
from queue import Queue
from threading import Event, Thread

import numpy as np
import torch
from tensorboard.program import TensorBoard
from torch.nn.functional import embedding
from torch.utils.tensorboard import SummaryWriter


class TorchTo(abc.ABC):

    def to(self, device: str | torch.device, dtype: torch.dtype = None, non_blocking: bool = False):
        device = torch.device(device) if isinstance(device, str) else device
        for key in self.__dict__:
            value = getattr(self, key)
            if isinstance(value, torch.Tensor):
                setattr(self, key, value.to(device=device, dtype=(dtype if (dtype is not None and dtype.is_floating_point and value.dtype.is_floating_point) else None), non_blocking=non_blocking).detach().clone().requires_grad_(value.requires_grad))
            elif isinstance(value, TorchTo):
                setattr(self, key, value.to(device=device, dtype=dtype, non_blocking=non_blocking))
        return self


@torch.compile
def fast_index(data: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return embedding(input=indices, weight=data.contiguous().view(data.shape[0], -1)).view(indices.shape + data.shape[1:])


@torch.compile
def inv_ex(x: torch.Tensor, inv: bool) -> torch.Tensor:
    return torch.linalg.inv_ex(x.to(torch.float32))[0].to(x.dtype) if inv else x


@contextmanager
def torch_no_compile():
    original_disable = torch._dynamo.config.disable
    original_verbose = torch._dynamo.config.verbose
    try:
        torch._dynamo.config.disable = True
        torch._dynamo.config.verbose = False
        print("[torch.compile] 已临时禁用")
        yield
    finally:
        torch._dynamo.config.disable = original_disable
        torch._dynamo.config.verbose = original_verbose
        print("[torch.compile] 已恢复原始状态")


class AsyncSummaryWriter:

    def __init__(self, log_dir):
        self.__writer = SummaryWriter(log_dir)
        self.__queue = Queue()
        self.__stop_event = Event()
        self.__worker = Thread(target=self.__run)
        self.__worker.daemon = True
        self.__worker.start()
        if os.environ.get('DISABLE_TENSORBOARD', '0') == '1':
            self.__tb = None
        else:
            self.__tb = TensorBoard()
            self.__tb.configure(argv=[None, '--logdir', log_dir, '--bind_all'])
            print(f"TensorBoard 已启动: {self.__tb.launch()}")

    def __run(self):
        try:
            while True:
                event, (name, values) = self.__queue.get()
                event.synchronize()
                try:
                    getattr(self.__writer, name)(**values)
                except Exception:
                    traceback.print_exc()
        except Exception:
            traceback.print_exc()
        finally:
            self.__writer.close()
            self.__writer = None

    def __put(self, data):
        event = torch.cuda.Event()
        event.record(stream=None)
        self.__queue.put((event, data))

    def add_scalar(self, tag: str, scalar_value: torch.Tensor | np.ndarray, global_step: int):
        if isinstance(scalar_value, torch.Tensor):
            scalar_value = scalar_value.detach().clone().to(device='cpu', non_blocking=True)
        self.__put(('add_scalar', dict(tag=tag, scalar_value=scalar_value, global_step=global_step)))

    def add_histogram(self, tag: str, values: torch.Tensor | np.ndarray, global_step: int):
        if values is None:
            return
        if isinstance(values, torch.Tensor):
            if values.numel() == 0:
                return
            values = values.detach().clone().to(device='cpu', non_blocking=True)
        if isinstance(values, np.ndarray) and values.size == 0:
            return
        self.__put(('add_histogram', dict(tag=tag, values=values, global_step=global_step)))
