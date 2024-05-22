"""Just some logging utilities to make logging easier and more consistent."""

import functools
import time
import contextvars
import sys
import logging

from loguru import logger

user_id_var = contextvars.ContextVar("user_id", default="anonymous")

logger.remove()

# additional log levels
try:
    logger.level("NOTICE", no=27, color="<green><b>", icon="x")
    logging.addLevelName(5, "TRACE")
    logging.addLevelName(25, "SUCCESS")
    logging.addLevelName(27, "NOTICE")
except TypeError as e:
    if "already exists" not in str(e):
        raise


# global logger configuration
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level> | <level>{extra}</level>",
    level="DEBUG",
    colorize=None,
)
logger.add("file_{time}.log", format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message} | {extra[user_id]}", level="DEBUG")


def get_bound_logger():
    return logger.bind(user_id=user_id_var.get())


def logger_wraps(*, entry=True, exit=True, level="DEBUG", timing=True):
    def decorator(func):
        name = func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger_ = logger.bind(user_id=user_id_var.get()).opt(depth=1)
            if entry:
                logger_.log(level, f"Entering '{name}' ({args=}, {kwargs=})")
            if timing:
                start = time.perf_counter()

            result = func(*args, **kwargs)

            if timing:
                end = time.perf_counter() - start
                logger_.log(level, "Function '{}' executed in {:.3f}s", name, end)

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


def main():
    user_id = "12345"
    token = user_id_var.set(user_id)
    logger_context = logger.bind(user_id=user_id)
    try:
        with logger_context.catch(reraise=False):
            g()
    finally:
        logger_context.info(f"Completed execution for user {user_id}")
        user_id_var.reset(token)
        print(f"{user_id_var.get()=}")


if __name__ == "__main__":
    main()
