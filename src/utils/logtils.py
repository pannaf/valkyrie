"""Just some logging utilities to make logging easier and more consistent."""

import functools
import time
import contextvars
import sys
import logging

from omegaconf import DictConfig
import hydra
import traceback

from loguru import logger

user_id_var = contextvars.ContextVar("user_id", default="anonymous")


def configure_logging(cfg: DictConfig):
    logger.remove()

    if cfg.logging.console.enable:
        logger.add(
            sys.stdout,
            format=cfg.logging.console.format,
            level=cfg.logging.console.level,
            colorize=cfg.logging.console.colorize,
        )

    if cfg.logging.file.enable:
        logger.add(
            cfg.logging.file.path,
            format=cfg.logging.file.format,
            level=cfg.logging.file.level,
        )


# additional log levels
try:
    logger.level("NOTICE", no=27, color="<green><b>", icon="x")
    logging.addLevelName(5, "TRACE")
    logging.addLevelName(25, "SUCCESS")
    logging.addLevelName(27, "NOTICE")
except TypeError as exc:
    if "already exists" not in str(exc):
        raise


def get_bound_logger():
    return logger.bind(user_id=user_id_var.get())


def logger_wraps(*, entry=True, exit=True, level="DEBUG", timing=True):
    def decorator(func):
        name = func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger_ = get_bound_logger().opt(depth=1)
            if entry:
                logger_.log(level, f"Entering '{name}' ({args=}, {kwargs=})")
            if timing:
                start = time.perf_counter()

            result = func(*args, **kwargs)

            if timing:
                end = time.perf_counter() - start
                logger_.log(level, f"Function '{name}' executed in {end}s")

            if exit:
                logger_.log(level, f"Exiting '{name}' ({result=})")
            return result

        return wrapper

    return decorator


class LoggingContextManager:
    def __init__(self, user_id):
        self.user_id = user_id
        self.token = None

    def __enter__(self):
        self.token = user_id_var.set(self.user_id)
        logger_context = logger.bind(user_id=self.user_id)
        return logger_context

    def __exit__(self, exc_type, exc_val, exc_tb):
        user_id_var.reset(self.token)


###-----------------###
# Usage example
###-----------------###


@logger_wraps(level="NOTICE")
def test_func(sleep_time=None):
    time.sleep(sleep_time)
    return "test"


@logger_wraps()
def f(x):
    return 100 / x


@logger_wraps()
def g():
    f(10)
    test_func(0.1)
    f(0)


@hydra.main(config_path="../../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    configure_logging(cfg)
    user_id = "12345"
    with LoggingContextManager(user_id) as logger_context:
        try:
            with logger_context.catch(reraise=False):
                g()
        finally:
            logger_context.info(f"Completed execution for user {user_id}")
            print(f"{user_id_var.get()=}")


if __name__ == "__main__":
    main()
