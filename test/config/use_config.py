from mmmcv.utils import Config

cfg = Config.fromfile('config-b.py')
print(cfg)
dict(a=1,
     b=dict(b1=[0, 1, 2], b2=None),
     c=(1, 2),
    d='string')