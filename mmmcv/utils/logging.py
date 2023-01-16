import logging

import torch.distributed as dist

logger_initialized: dict = {}


def get_logger(name, log_file=None, log_level=logging.INFO, file_mode='w'):
    """Initialize and get a logger by name.

    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified and the process rank is 0, a FileHandler
    will also be added.

    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.
        file_mode (str): The file mode used in opening log file.
            Defaults to 'w'.

    Returns:
        logging.Logger: The expected logger.
    初始化并获取一个名称的日志。

    如果日志记录器没有被初始化，此方法将初始化
    通过添加一个或两个处理程序来记录，否则初始化记录器将
    被直接退回 在初始化过程中，StreamHandler 始终是
    补充道。 如果指定了 'log_file'，并且进程的秩为 0，则一个 FileHandler
    亦将会增加。

   Args:
        name(str): 日志名。
        log_file (str|None): 日志文件名。 如果指定，文件处理程序
            将会被添加到记录器中。
        log_level (int): 记录器级别。 请注意,只有以下过程:
            等级0受到影响,而其他过程会将级别设定为
            因此，"错误"在大多数时候是沉默的。
        file_mode (str): 用于打开日志文件的文件模式。
            默认值为"w"。

   Returns:
        logging.Logger:预期的日志记录器。
    """
    logger = logging.getLogger(name)
    #检查 logger_initialized 这个列表中是否已经存在 name，如果存在，直接返回该 logger。
    if name in logger_initialized:
        return logger
    # handle hierarchical names 处理层次名称
    # e.g., logger "a" is initialized, then logger "a.b" will skip the
    # initialization since it is a child of "a".
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    # handle duplicate logs to the console 处理重复日志到控制台
    # Starting in 1.8.0, PyTorch DDP attaches a StreamHandler <stderr> (NOTSET)
    # to the root logger. As logger.propagate is True by default, this root
    # level handler causes logging messages from rank>0 processes to
    # unexpectedly show up on the console, creating much unwanted clutter.
    # To fix this issue, we set the root logger's StreamHandler, if any, to log
    # at the ERROR level.
    # 从 1.8.0 开始，PyTorch DDP 附加了一个 StreamHandler <stderr> (NOTSET)
    # 到根记录器。 由于 logger.propagate 在缺省情况下为 True，所以这个根目录是
    # 级别处理程序导致日志消息从秩>0 进程到
    # 意外地出现在控制台上，造成许多不必要的混乱。
    # 为了解决这个问题，我们将根记录器的 StreamHandler（如果有的话）设置为 log
    # 在错误级别。
    for handler in logger.root.handlers:
        if type(handler) is logging.StreamHandler:
            handler.setLevel(logging.ERROR)

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # only rank 0 will add a FileHandler
    if rank == 0 and log_file is not None:
        # Here, the default behaviour of the official logger is 'a'. Thus, we
        # provide an interface to change the file mode to the default
        # behaviour.
        # 在这里，官方日志记录器的默认行为是 'a'。 因此，我们
        # 提供一个界面来改变文件模式到默认值
        # 行为。
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    logger_initialized[name] = True

    return logger

def print_log(msg, logger=None, level=logging.INFO):
    """Print a log message.

    Args:
        msg (str): The message to be logged.
        logger (logging.Logger | str | None): The logger to be used.
            Some special loggers are:

            - "silent": no message will be printed.
            - other str: the logger obtained with `get_root_logger(logger)`.
            - None: The `print()` method will be used to print log messages.
        level (int): Logging level. Only available when `logger` is a Logger
            object or "root".
    打印日志消息。

    Args:
        msg (str):要记录的消息。
         logger (logging.Logger | str | None): 要使用的日志记录器。
            一些特别的伐木者包括:

            -"silent":不打印任何信息。
            - 其他 str:使用 'get_root_select(select)' 获得的日志记录器
            - none: 'print()' 方法将用于打印日志消息。
        level(int): logging水平。 只有在'logger'是logger时才能使用
            对象或"根"
    """
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger == 'silent':
        pass
    elif isinstance(logger, str):
        _logger = get_logger(logger)
        _logger.log(level, msg)
    else:
        raise TypeError(
            'logger should be either a logging.Logger object, str, '
            f'"silent" or None, but got {type(logger)}')