import mmmcv
import torch

def is_mlu_available():
    """Returns a bool indicating if MLU is currently available.
    hasattr is equal like bools
    """
    return hasattr(torch, 'is_mlu_available') and torch.is_mlu_available()

def get_device():
    """Returns an available device, cpu, cuda or mlu.
    首先，它创建了一个名为is_device_available的字典，其中包含了 CUDA 和 MLU 设备是否可用的布尔值。
    然后，它使用列表推导式创建了一个名为device_list的列表，该列表包含了所有可用设备的名称。
    最后，如果device_list只包含一个元素，该函数返回该元素，否则返回'cpu'。
    """
    is_device_available = {
        'cuda': torch.cuda.is_available(),
        'mlu': is_mlu_available()
    }
    device_list = [k for k, v in is_device_available.items() if v]
    return device_list[0] if len(device_list) == 1 else 'cpu'
