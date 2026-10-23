# By Hanrong Ye
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformers.decoder import B3Decoder
import pdb

from easydict import EasyDict as edict
INTERPOLATE_MODE = 'bilinear'


class B3Net(nn.Module):
    def __init__(self, p, backbone, backbone_channels, heads):
        super(B3Net, self).__init__()
        self.tasks = p.TASKS.NAMES

        self.backbone = backbone
        self.multi_task_decoder = B3Decoder(p)
        self.heads = heads

    def forward(self, x):
        img_size = x.size()[-2:]
        out = {}

        x, selected_fea = self.backbone(x)

        task_features, inter_preds = self.multi_task_decoder(selected_fea)

        out = task_features
        for t in self.tasks:
            out[t] = F.interpolate(self.heads[t](task_features[t]), img_size, mode=INTERPOLATE_MODE)
        out['inter_preds'] = {t: F.interpolate(v, img_size, mode=INTERPOLATE_MODE) for t, v in inter_preds.items()}

        return out
