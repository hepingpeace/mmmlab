# Copyright (c) OpenMMLab. All rights reserved.
from mmmcv.utils import collect_env as collect_base_env
from mmmcv.utils import get_git_hash

import mmmseg


def collect_env():
    """Collect the information of the running environments.
    该函数的作用是收集运行环境的信息。
    首先调用 collect_base_env() 函数来收集基本的环境信息，然后在结果字典中添加一个名为 'MMSegmentation' 的键值对，
    其值为 'mmseg.version + get_git_hash()[:7]' 。'mmseg.version'表示库的版本号，
    'get_git_hash()[:7]'表示代码的git版本号。最后返回这个字典。
    """
    env_info = collect_base_env()
    env_info['MMSegmentation'] = f'{mmmseg.__version__}+{get_git_hash()[:7]}'

    return env_info


if __name__ == '__main__':
    for name, val in collect_env().items():
        print('{}: {}'.format(name, val))
