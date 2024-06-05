from src.utils.logtils import logger_wraps, get_bound_logger


@logger_wraps(level="NOTICE")
def test_func(sleep_time=None):
    logger_ = get_bound_logger()
    logger_.info(f"test_func started {sleep_time=}")
    for _ in range(1000000):
        pass
    logger_.info("test_func executed")
    return "test"


@logger_wraps()
def f(x):
    return 100 / x


@logger_wraps()
def g():
    f(10)
    test_func(20)
    f(20)
    if 1:
        f(0)
