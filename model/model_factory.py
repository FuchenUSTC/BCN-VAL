# -*- encoding: utf-8 -*-
# auth: Fuchen Long
# mail: longfc.ustc@gmail.com
# date: 2021/09/28
# desc: model factory for registration

import sys
import numpy as np
import torch

model_dict = {}
transfer_dict = {}


def get_model_by_name(net_name, **kwargs):
    return model_dict.get(net_name)(**kwargs)


# model registration
# from https://github.com/rwightman/pytorch-image-models/
def register_model(fn):
    mod = sys.modules[fn.__module__]
    model_name = fn.__name__

    # add entries to registry dict/sets
    assert model_name not in model_dict
    model_dict[model_name] = fn
    if hasattr(mod, 'transfer_weights'):
        transfer_dict[model_name] = mod.transfer_weights
    else:
        transfer_dict[model_name] = None
    return fn
