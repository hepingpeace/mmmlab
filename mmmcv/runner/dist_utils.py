import functools
import os
import socket
import subprocess
from collections import OrderedDict
from typing import Callable, List, Optional, Tuple

import torch
import torch.multiprocessing as mp
from torch import distributed as dist
# from torch._utils import (_flatten_dense_tensors, _take_tensors,
#                           _unflatten_dense_tensors)

from mmmcv.utils import IS_MLU_AVAILABLE, IS_NPU_AVAILABLE

def init_dist(launcher: str, backend: str = 'nccl', **kwargs) -> None:
    """
    该函数有两个参数，分别是 launcher 和 backend，其中 launcher 的默认值为 'nccl'，后面还有一些关键字参数。

    该函数的作用是根据传入的 launcher 参数来初始化分布式训练。
    如果 launcher 为 'pytorch'，则调用 _init_dist_pytorch 函数来初始化分布式训练；
    如果 launcher 为 'mpi'，则调用 _init_dist_mpi 函数来初始化分布式训练；
    如果 launcher 为 'slurm'，则调用 _init_dist_slurm 函数来初始化分布式训练；
    如果 launcher 为其他值，则抛出 ValueError 异常。
    """
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    if launcher == 'pytorch':
        _init_dist_pytorch(backend, **kwargs)
    elif launcher == 'mpi':
        _init_dist_mpi(backend, **kwargs)
    elif launcher == 'slurm':
        _init_dist_slurm(backend, **kwargs)
    else:
        raise ValueError(f'Invalid launcher type: {launcher}')

def get_dist_info() -> Tuple[int, int]:
    """
    该函数没有参数，并返回一个元组，其中包含两个整数。
    该函数的作用是获取分布式训练的信息。
    首先，通过调用 dist.is_available() 和 dist.is_initialized() 函数来检查分布式训练是否可用和是否已经初始化。
    如果可用且已初始化，则调用 dist.get_rank() 和 dist.get_world_size() 函数来获取当前进程的 rank 和 world_size。
    否则将 rank 和 world_size 设置为 0 和 1。最后返回 rank 和 world_size。
    """
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size

def _init_dist_pytorch(backend: str, **kwargs) -> None:
    # TODO: use local_rank instead of rank % num_gpus
    """
    它用于初始化分布式训练环境。它首先检查是否有 MLU 或 NPU 可用，如果有则使用相应的库并将设备设置为当前进程的 rank。
    如果没有 MLU 或 NPU，则使用 CUDA 并将设备设置为当前进程的 rank。
    最后调用 dist.init_process_group() 初始化分布式训练环境。
    """
    rank = int(os.environ['RANK'])
    if IS_MLU_AVAILABLE:
        import torch_mlu  # noqa: F401
        torch.mlu.set_device(rank)
        dist.init_process_group(
            backend='cncl',
            rank=rank,
            world_size=int(os.environ['WORLD_SIZE']),
            **kwargs)
    elif IS_NPU_AVAILABLE:
        import torch_npu  # noqa: F401
        num_npus = torch.npu.device_count()
        torch.npu.set_device(rank % num_npus)
        dist.init_process_group(
            backend='hccl',
            rank=rank,
            world_size=int(os.environ['WORLD_SIZE']),
            **kwargs)
    else:
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(rank % num_gpus)
        dist.init_process_group(backend=backend, **kwargs)

def _init_dist_mpi(backend: str, **kwargs) -> None:
    local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    if 'MASTER_PORT' not in os.environ:
        # 29500 is torch.distributed default port
        os.environ['MASTER_PORT'] = '29500'
    if 'MASTER_ADDR' not in os.environ:
        raise KeyError('The environment variable MASTER_ADDR is not set')
    os.environ['WORLD_SIZE'] = os.environ['OMPI_COMM_WORLD_SIZE']
    os.environ['RANK'] = os.environ['OMPI_COMM_WORLD_RANK']
    dist.init_process_group(backend=backend, **kwargs)

def _init_dist_slurm(backend: str, port: Optional[int] = None) -> None:
    """Initialize slurm distributed training environment.
    If argument ``port`` is not specified, then the master port will be system
    environment variable ``MASTER_PORT``. If ``MASTER_PORT`` is not in system
    environment variable, then a default port ``29500`` will be used.
    Args:
        backend (str): Backend of torch.distributed.
        port (int, optional): Master port. Defaults to None.
        初始化混浊分布式训练环境。
    如果没有指定参数 'port''，那么主端口将是 system
    环境变量"MASTER_PORT"。 如果"MASTER_PORT"不在系统内
    如果使用环境变量,则将使用默认端口"29500"。
    Args:
        后端(str): torch.distributed的后端.
        端口（int，可选）: 主港口 默认值为"无"

    """
    # os.environ是一个字典对象，它存储了系统环境变量。您可以访问环境变量的值，
    # 例如os.environ['SLURM_PROCID']可以获取当前进程的ID。
    # 您还可以修改或删除环境变量的值，例如os.environ['SLURM_PROCID'] = "new_value"。
    # 修改会影响到后续程序的运行，删除会删除该环境变量。
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    # 使用了 subprocess.getoutput() 方法来执行一个 shell 命令，其中传入了一个字符串变量 node_list。
    # 这个命令会在 SLURM 集群中查找主机名，并使用 scontrol show hostname 命令来显示。
    # 输出会经过 head -n1 处理只显示第一行，最终结果会被保存在变量 addr 中
    addr = subprocess.getoutput(
        f'scontrol show hostname {node_list} | head -n1')
    # specify master port
    if port is not None:
        os.environ['MASTER_PORT'] = str(port)
    elif 'MASTER_PORT' in os.environ:
        pass  # use MASTER_PORT in the environment variable
    else:
        # if torch.distributed default port(29500) is available
        # then use it, else find a free port
        if _is_free_port(29500):
            os.environ['MASTER_PORT'] = '29500'
        else:
            os.environ['MASTER_PORT'] = str(_find_free_port())
    # use MASTER_ADDR in the environment variable if it already exists
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend=backend)

def _is_free_port(port: int) -> bool:
    """

    用来检查一个给定的端口是否可用。
    首先使用 socket.gethostbyname_ex() 方法获取当前主机的 IP 地址列表，并将 'localhost' 添加到列表中。
    接着，使用 socket.socket() 创建一个 TCP 套接字。
    最后，使用 s.connect_ex((ip, port)) != 0 连接每个 IP 地址的给定端口，如果连接失败，则表示该端口可用。
    如果所有的IP都不能连接，则返回True，表示端口可用。

    """
    ips = socket.gethostbyname_ex(socket.gethostname())[-1]
    ips.append('localhost')
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return all(s.connect_ex((ip, port)) != 0 for ip in ips)

def _find_free_port() -> str:
    """
    它用来查找一个可用的端口。
    首先，使用 socket.socket() 创建一个 TCP 套接字。
    接着，使用 sock.bind(('', 0)) 绑定到端口 0。这将导致操作系统为我们找到一个可用的端口。
    最后，使用 sock.getsockname()[1] 获取当前绑定的端口号，并关闭套接字。并返回该端口号。
    注意：尽管如此，这个端口仍有可能被其他进程占用。

    """
    # Copied from https://github.com/facebookresearch/detectron2/blob/main/detectron2/engine/launch.py # noqa: E501
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port