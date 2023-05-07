
import numpy as np
import torch
from torch.utils.data import Dataset
from e3nn import o3

from torch import Tensor
from typing import *


class NeighborhoodsDataset(Dataset):
    def __init__(self, x: Tensor, irreps: o3.Irreps, y: Tensor, c: List):
        self.x = x # [N, dim]
        self.y = y # [N,]
        self.c = c # [N, ANY]
        assert x.shape[0] == y.shape[0]

        self.ls_indices = torch.cat([torch.tensor([l]).repeat(2*l+1) for l in irreps.ls])
        self.unique_ls = sorted(list(set(irreps.ls)))
    
    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, idx: int):
        x_fiber = {}
        for l in self.unique_ls:
            x_fiber[l] = self.x[idx][self.ls_indices == l].view(-1, 2*l+1)
        
        return x_fiber, self.x[idx], self.y[idx], self.c[idx]

