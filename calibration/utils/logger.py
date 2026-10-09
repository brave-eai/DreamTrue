import logging
import os
import types
from typing import Type, Mapping

# pycharm 编辑器 -> 日志高亮显示
#   消息模式：^(\d{4}\-\d{2}\-\d{2} \d{2}\:\d{2}\:\d{2}) *\:\: *([\w\.\-]+):(\d+) *\:\: *([\w\.\-]+) *\:\: *([\w]+) *\:\: (.*?)$
#   消息开始模式：^(\d{4}\-\d{2}\-\d{2} \d{2}\:\d{2}\:\d{2})
#   时间格式：yyyy-MM-dd HH:mm:ss
#   时间捕获组：1
#   严重性捕获组：4
#   类别捕获组：3


class ColorLogger:

    def __init__(self, logger_name: str = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        long_name = logger_name or (self.__class__.__name__ + '.' + str(id(self)))
        short_name = logger_name or self.__class__.__name__
        self._logger = logging.getLogger(long_name)
        ln = short_name
        if len(ln) >= 18:
            ln = ln[:8] + '..' + ln[-8:]
        self._logger.name = ln
        self._logger.propagate = False
        self._lprefix = ''
        if len(self._logger.handlers) == 0:
            level_main, level_ext = _get_hydra_level(short_name)
            filter_main, filter_ext = LevelFilter(level_main), LevelFilter(level_ext)
            plain_f = logging.Formatter(
                "{asctime} :: {filename:>12.12}:{lineno:<4} :: {name:^18.18} :: {levelname:^8.8} :: {message}",
                datefmt="%Y-%m-%d %H:%M:%S",
                style="{",
            )
            try:
                from colorlog import ColoredFormatter

                color_f = ColoredFormatter(
                    "{green}{asctime}{reset} :: {bold_blue}{filename:>12.12}:{lineno:<4}{reset} :: {bold_purple}{name:^18.18}{reset} :: {log_color}{levelname:^8.8}{reset} :: {bold_white}{message}{reset}",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    reset=True,
                    log_colors={
                        "INFO": "bold_cyan",
                        "DEBUG": "thin_yellow",
                        "WARNING": "bold_yellow",
                        "ERROR": "bold_red",
                        "CRITICAL": "bold_red,bg_white",
                    },
                    style="{",
                )
            except ImportError:
                color_f = plain_f
            handler = logging.StreamHandler()
            handler.addFilter(filter_main)
            handler.setFormatter(color_f)
            self._logger.addHandler(handler)
            try:
                from hydra.core.hydra_config import HydraConfig
                out_dir = HydraConfig.get().runtime.output_dir
            except (ValueError, ImportError):
                out_dir = None
            if isinstance(out_dir, str):
                os.makedirs(os.path.join(out_dir, 'logs'), exist_ok=True)
                handler_f = logging.FileHandler(os.path.join(out_dir, 'logs', 'log.log'), mode="a")
                handler_f.addFilter(filter_main)
                handler_f.setFormatter(plain_f)
                self._logger.addHandler(handler_f)
                if color_f != plain_f:
                    handler_f2 = logging.FileHandler(os.path.join(out_dir, 'logs', 'log.clog'), mode="a")
                    handler_f2.addFilter(filter_main)
                    handler_f2.setFormatter(color_f)
                    self._logger.addHandler(handler_f2)
                if level_ext != logging.NOTSET:
                    handler_f3 = logging.FileHandler(os.path.join(out_dir, 'logs', f'{long_name}.log'), mode="a")
                    handler_f3.addFilter(filter_ext)
                    handler_f3.setFormatter(plain_f)
                    self._logger.addHandler(handler_f3)
            if level_ext != logging.NOTSET:
                self._logger.setLevel(min(level_main, level_ext))
            else:
                self._logger.setLevel(level_main)

    def exception(
        self, msg: object, exc_info: None | bool | tuple[Type[BaseException], BaseException, types.TracebackType | None] | tuple[None, None, None] | BaseException = True, stack_info: bool = False, stacklevel: int = 1,
        extra: Mapping[str, object] | None = None
    ) -> None:
        if self._logger.isEnabledFor(logging.ERROR):
            self._logger._log(logging.ERROR, self._lprefix + str(msg), args=(), exc_info=exc_info, stack_info=stack_info, stacklevel=stacklevel + 1, extra=extra)

    def warning(
        self, msg: object, exc_info: None | bool | tuple[Type[BaseException], BaseException, types.TracebackType | None] | tuple[None, None, None] | BaseException = None, stack_info: bool = False, stacklevel: int = 1,
        extra: Mapping[str, object] | None = None
    ) -> None:
        if self._logger.isEnabledFor(logging.WARNING):
            self._logger._log(logging.WARNING, self._lprefix + str(msg), args=(), exc_info=exc_info, stack_info=stack_info, stacklevel=stacklevel + 1, extra=extra)

    def debug(
        self, msg: object, exc_info: None | bool | tuple[Type[BaseException], BaseException, types.TracebackType | None] | tuple[None, None, None] | BaseException = None, stack_info: bool = False, stacklevel: int = 1,
        extra: Mapping[str, object] | None = None
    ) -> None:
        if self._logger.isEnabledFor(logging.DEBUG):
            self._logger._log(logging.DEBUG, self._lprefix + str(msg), args=(), exc_info=exc_info, stack_info=stack_info, stacklevel=stacklevel + 1, extra=extra)

    def info(
        self, msg: object, exc_info: None | bool | tuple[Type[BaseException], BaseException, types.TracebackType | None] | tuple[None, None, None] | BaseException = None, stack_info: bool = False, stacklevel: int = 1,
        extra: Mapping[str, object] | None = None
    ) -> None:
        if self._logger.isEnabledFor(logging.INFO):
            self._logger._log(logging.INFO, self._lprefix + str(msg), args=(), exc_info=exc_info, stack_info=stack_info, stacklevel=stacklevel + 1, extra=extra)

    def log(
        self, level: int, msg: object, exc_info: None | bool | tuple[Type[BaseException], BaseException, types.TracebackType | None] | tuple[None, None, None] | BaseException = None, stack_info: bool = False, stacklevel: int = 1,
        extra: Mapping[str, object] | None = None
    ) -> None:
        if self._logger.isEnabledFor(level):
            self._logger._log(level, self._lprefix + str(msg), args=(), exc_info=exc_info, stack_info=stack_info, stacklevel=stacklevel + 1, extra=extra)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None or exc_val is not None or exc_tb is not None:
            self.exception(msg='', exc_info=(exc_type, exc_val, exc_tb), stacklevel=2)
        return False


def _get_hydra_level(name: str) -> tuple[int, int]:
    # return logging.DEBUG, logging.DEBUG
    return logging.INFO, logging.INFO


class LevelFilter(logging.Filter):

    def __init__(self, level: int):
        super().__init__()
        self.__level = level

    def filter(self, record):
        return record.levelno >= self.__level
