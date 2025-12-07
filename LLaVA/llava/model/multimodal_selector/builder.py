import torch
import torch.nn as nn
import re
import os
from .selector import Selector, SelectorForClassification


def build_vision_selector(mm_selector_cfg, **kwargs):

    mm_selector = getattr(mm_selector_cfg, 'mm_selector', getattr(mm_selector_cfg, 'mm_selector', None))

    return SelectorForClassification(config=mm_selector_cfg, **kwargs)

