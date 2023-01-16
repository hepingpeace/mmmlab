import os
import random
import warnings

import mmmcv
import numpy as np
import torch

def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.
    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.
    Returns:
        int: Seed to be used.
    初始化随机种子。

    如果不设置种子，种子将自动随机化，
    然后广播到所有进程，以防止一些潜在的错误。
    论点:
        种子（int，可选）: 种子 默认设置为"无"。
        设备(str): 播种器种子的装置
            默认值为"cuda"。
    Returns:
        int: 使用的种子。
    """
    if seed is not None:
        return seed

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    rank, world_size = get_dist_info()
    seed = np.random.randint(2**31)
    if world_size == 1:
        return seed