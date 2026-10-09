import atexit
import multiprocessing
import os
import traceback
from concurrent.futures import as_completed
from dataclasses import dataclass
from multiprocessing.queues import Queue as QueueType
from multiprocessing.synchronize import Lock as LockType
from typing import Iterable

from tqdm import tqdm


def wait_futures(futures: list, max_len: int, callback=None):
    while True:
        as_completed(futures)
        tmp_future = []
        for f in futures:
            if not f.done():
                tmp_future.append(f)
            elif callback:
                callback(f.result())
        futures = tmp_future
        if len(futures) <= max_len:
            break
    return futures


@dataclass
class HeavyWorkerInfo:
    tqdm_lock: LockType
    tqdm_position: int
    tqdm_position_queue: QueueType
    gpu_device: int
    gpu_device_queue: QueueType

    def pop(self):
        self.tqdm_position = self.tqdm_position_queue.get()
        self.gpu_device = self.gpu_device_queue.get()

    def push(self):
        self.tqdm_position_queue.put(self.tqdm_position)
        self.gpu_device_queue.put(self.gpu_device)

    @classmethod
    def new(cls, max_workers: int, gpu_device_count: int):
        context = multiprocessing.get_context('spawn')
        info = cls(
            tqdm_lock=context.Lock(),
            tqdm_position=-1,
            tqdm_position_queue=context.Queue(),
            gpu_device=-1,
            gpu_device_queue=context.Queue(),
        )
        tqdm.set_lock(info.tqdm_lock)
        for i in range(max_workers * max(1, gpu_device_count)):
            info.tqdm_position_queue.put(i + 1)
        for i in range(max_workers):
            if gpu_device_count == 0:
                info.gpu_device_queue.put(-1)
            else:
                for gpu_id in range(gpu_device_count):
                    info.gpu_device_queue.put(gpu_id)
        return info, context


class HeavyWorker():
    info: HeavyWorkerInfo | None = None

    @classmethod
    def init(cls, info: HeavyWorkerInfo):
        try:
            atexit.register(cls.cleanup)
            cls.info = info
            tqdm.set_lock(cls.info.tqdm_lock)
            cls.info.pop()
        except Exception as e:
            traceback.print_exc()
            raise e

    @classmethod
    def cleanup(cls):
        try:
            if cls.info is not None:
                cls.info.push()
                cls.info = None
        except Exception as e:
            traceback.print_exc()

    @classmethod
    def tqdm(cls, iterable: Iterable, desc: str | None = None):
        assert cls.info is not None, 'Info not initialized'
        return tqdm(iterable, desc=f'{cls.info.tqdm_position:02d}|{os.getpid():8d}|GPU{cls.info.gpu_device}' + (('|' + desc[-30:]) if desc else ''), mininterval=0, position=cls.info.tqdm_position, leave=False, ncols=100)
