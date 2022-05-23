"""Utilities for parallel runs"""
import logging
import multiprocessing.shared_memory as shm
from datetime import timedelta
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.pool import Pool
from time import time
from typing import Callable, Iterable, Generator, Tuple

import numpy as np


class RunPipeline:
    """Wrapper class to run the pipeline in a process pool

    - obj: the pipeline object (e.g. self)
    - param_name: the name of the parameter/property to load the pyterrier transformer"""
    def __init__(self, obj, param_name):
        self.obj = obj
        self.param_name = param_name
        self.pipeline = None

    def __call__(self, q):
        if self.pipeline is None:
            self.pipeline = getattr(self.obj, self.param_name)
        return self.pipeline(q)


def _time_wrapper(func, args, kwargs):
    st = time()
    r = func(*args, **kwargs)
    _time = timedelta(seconds=(time() - st))
    return _time, r


class TimeLogWrapper:
    def __init__(self, func: Callable, msg_format: str, logger_name: str):
        """Wrapper that log the execution time.
        The returned value is (timedelta, original_result).

        :param func: function or callable to run
        :param msg_format: format string for the log message
        :param logger_name: name of the logger"""
        self.func = func
        self.msg_format = msg_format
        self.logger_name = logger_name

    def __callback(self, res):
        logger = logging.getLogger(self.logger_name)
        logger.info(self.msg_format, res[0])

    def __error_callback(self, ex):
        logger = logging.getLogger(self.logger_name)
        logger.exception('Failed to execute background job', exc_info=ex)

    def run(self, pool: Pool, args: Iterable = None, kwargs: dict = None):
        args = tuple() if args is None else args
        kwargs = {} if kwargs is None else kwargs
        return pool.apply_async(_time_wrapper, (self.func, args, kwargs),
                                callback=self.__callback, error_callback=self.__error_callback)


class RunWrapper:
    def __init__(self, obj, method: str):
        self.obj = obj
        self.method = method

    def __call__(self, *args, **kwargs):
        return getattr(self.obj, self.method)(*args, **kwargs)


class BufferedGenerator:
    def __init__(self, gen: Generator, max_queue_size: int):
        """Wrapper that buffer a generator using a thread.

        :param gen: generator to buffer
        :param max_queue_size: max size to buffer"""
        from threading import Thread
        from queue import Queue
        self.__generator = gen
        self.__thread = Thread(target=self.__enqueue)
        self.__queue = Queue(max_queue_size)
        self.__END = object()

        self.__thread.start()

    def __enqueue(self):
        for item in self.__generator:
            self.__queue.put(item)
        self.__queue.put(self.__END)

    def __next__(self):
        item = self.__queue.get()
        if item is self.__END:
            raise StopIteration
        return item

    def __iter__(self):
        return self


class LockedObject:
    def __init__(self, obj, mutex, methods: list = None):
        self.__obj = obj
        self.__mutex = mutex
        self.__methods = methods
        if methods is not None:
            for m in methods:
                setattr(self, m, lambda *args, **kwargs: self.__call_method(m, *args, **kwargs))

    def __call_method(self, method, *args, **kwargs):
        m = getattr(self.__obj, method)
        self.__mutex.acquire()
        r = m(*args, **kwargs)
        self.__mutex.release()
        return r

    def __call__(self, *args, **kwargs):
        self.__mutex.acquire()
        r = self.__obj(*args, **kwargs)
        self.__mutex.release()
        return r

    def __getstate__(self):
        return {'obj': self.__obj, 'mutex': self.__mutex, 'methods': self.__methods}

    def __setstate__(self, state):
        self.__obj = state['obj']
        self.__mutex = state['mutex']
        self.__methods = state['methods']
        if self.__methods is not None:
            for m in self.__methods:
                setattr(self, m, lambda *args, **kwargs: self.__call_method(m, *args, **kwargs))


def np_array_to_shm(array: np.ndarray, smm: SharedMemoryManager) -> Tuple[shm.SharedMemory, np.ndarray]:
    shm_buffer = smm.SharedMemory(array.nbytes)
    shm_array = np.ndarray(array.shape, dtype=array.dtype, buffer=shm_buffer.buf)
    shm_array[:] = array[:]
    return shm_buffer, shm_array

