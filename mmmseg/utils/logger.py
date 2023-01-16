import logging

from mmmcv.utils import get_logger


def get_root_logger(log_file=None, log_level=logging.INFO):
    """
    Get the root logger.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added. The name of the root logger is the top-level package name,
    e.g., "mmseg".

    Args:
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.

    Returns:
        logging.Logger: The root logger.
    获取根记录器。

    如果日志记录器没有被初始化，它将被初始化。 缺省情况下a
    将添加流处理程序。 如果指定了 'log_file'，则 FileHandler 将执行以下操作:
    另加。 根记录器的名称是top-level package name，
    例如「mmseg」。

    Args:
        log_file (str|无): 日志文件名。 如果指定，文件处理程序
            将添加到根记录器中。
        log_level (int): 根记录器级别。 请注意,只有以下过程:
            等级0受到影响,而其他过程则会把等级设定为
            "错误"，并且大部分时间保持沉默。

   Returns:
       logging.Logger: 根记录器。
    """

    logger = get_logger(name='mmseg', log_file=log_file, log_level=log_level)

    return logger