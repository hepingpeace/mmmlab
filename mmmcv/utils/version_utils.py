# Copyright (c) OpenMMLab. All rights reserved.
import os
import subprocess
import warnings

from packaging.version import parse

def get_git_hash(fallback='unknown', digits=None):
    """Get the git hash of the current repo.

    Args:
        fallback (str, optional): The fallback string when git hash is
            unavailable. Defaults to 'unknown'.
        digits (int, optional): kept digits of the hash. Defaults to None,
            meaning all digits are kept.

    Returns:
        str: Git commit hash.
    首先，它检查参数 digits 是否是整数。如果不是，则抛出 TypeError。
    接着，它使用 _minimal_ext_cmd(['git', 'rev-parse', 'HEAD']) 调用 git 命令获取当前仓库的 hash 值。
    如果指定了 digits 参数，则只保留前 digits 位。
    如果调用 git 命令时出现 OSError，则返回 fallback 参数的值。
    最后返回 git hash 值.

    """

    if digits is not None and not isinstance(digits, int):
        raise TypeError('digits must be None or an integer')

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        sha = out.strip().decode('ascii')
        if digits is not None:
            sha = sha[:digits]
    except OSError:
        sha = fallback

    return sha


def _minimal_ext_cmd(cmd):
    # construct minimal environment
    env = {}
    for k in ['SYSTEMROOT', 'PATH', 'HOME']:
        v = os.environ.get(k)
        if v is not None:
            env[k] = v
    # LANGUAGE is used on win32
    env['LANGUAGE'] = 'C'
    env['LANG'] = 'C'
    env['LC_ALL'] = 'C'
    out = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
    return out