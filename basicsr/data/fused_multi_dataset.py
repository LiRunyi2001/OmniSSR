import functools
import random
import math
from PIL import Image

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode
from torchvision import transforms
from timm.models.layers import to_2tuple

from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.data.__init__ import build_dataset

@DATASET_REGISTRY.register()
class FusedMultiDataset(Dataset):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.datasets = []
        self.make_dataset()
        self.repeat = self.opt.get('repeat', 1)  # train: 100, val: 1, test: 1
        # calculate dataset length
        self.length = sum([len(d) for d in self.datasets])


    def make_dataset(self):
        import datasets
        for dataset_key in self.opt['datasets']:
            dataset_opt = self.opt['datasets'][dataset_key]
            dataset = build_dataset(dataset_opt)
            self.datasets.append(dataset)

    def __len__(self):
        return self.length * self.repeat
    
    def __getitem__(self, idx):
        idx = idx % self.length

        for dataset in self.datasets:
            if idx < len(dataset):
                break
            idx -= len(dataset)

        return dataset.__getitem__(idx)

