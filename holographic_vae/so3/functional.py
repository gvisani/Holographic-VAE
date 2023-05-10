

import numpy as np
import torch
from torch import Tensor
import e3nn
from e3nn import o3

from typing import *

def get_random_wigner_D(lmax: int):
    rot_matrix = o3.rand_matrix()
    alpha, beta, gamma = o3.matrix_to_angles(rot_matrix)
    wigner = {}
    for l in range(lmax + 1):
        wigner[l] = o3.wigner_D(l, alpha, beta, gamma)
    return wigner

def get_wigner_D_from_rot_matrix(lmax: int, rot_matrix: Tensor):
    alpha, beta, gamma = o3.matrix_to_angles(rot_matrix)
    wigner = {}
    for l in range(lmax + 1):
        wigner[l] = o3.wigner_D(l, alpha, beta, gamma)
    return wigner

def get_wigner_D_from_alpha_beta_gamma(lmax: int, alpha: Tensor, beta: Tensor, gamma: Tensor):
    wigner = {}
    for l in range(lmax + 1):
        wigner[l] = o3.wigner_D(l, alpha, beta, gamma)
    return wigner

def rotate_signal(signal: Tensor, irreps: o3.Irreps, wigner: Dict):
    '''
    wigner must contain wigner-D matrices for all l's in irreps, otherwise a KeyError will be thrown
    '''
    wigner_ls = [wigner[l] for l in irreps.ls]
    rot_mat = torch.block_diag(*wigner_ls)
    rotated_signal = torch.matmul(signal, torch.t(rot_mat)) # Compute R = S * W^T --> 
    return rotated_signal


def get_vectors_at_l(signal: Tensor, irreps: o3.Irreps, l: int):
    ls_indices = torch.cat([torch.tensor(irreps.ls)[torch.tensor(irreps.ls) == l].repeat(2*l+1) for l in sorted(list(set(irreps.ls)))])
    return signal[:, ls_indices == l]

def filter_by_l(tensors, irreps, lmax):
    ls_indices = torch.cat([torch.tensor([l]).repeat(2*l+1) for l in irreps.ls])
    return tensors[:, ls_indices <= lmax]

def filter_by_channels(tensors, mul_rst, channel_idxs):
    separated_tensors = mul_rst.separate(tensors)
    new_tensors = []
    for ch in channel_idxs:
        new_tensors.append(separated_tensors[:, ch, :])
    
    filtered_tensors = mul_rst.combine(torch.stack(new_tensors, dim=1))

    return filtered_tensors

def make_vec(signals: Dict[int, Tensor]):
    vec = []
    batch_size = signals[0].shape[0]
    for l in signals:
        vec.append(signals[l].reshape(batch_size, -1))
    
    return torch.cat(vec, dim=-1)

def make_dict(signals: Tensor, irreps: o3.Irreps):
    ls_indices = torch.cat([torch.tensor([l]).repeat(2*l+1) for l in irreps.ls])
    signals_dict = {}
    for l in sorted(list(set(irreps.ls))):
        signals_dict[l] = signals[:, ls_indices == l].reshape(signals.shape[0], -1, 2*l+1)
    return signals_dict

def put_dict_on_device(signals_dict: Dict[int, Tensor], device: str):
    for l in signals_dict:
        signals_dict[l] = signals_dict[l].float().to(device)
    return signals_dict

def take_dict_down_from_device(signals_dict: Dict[int, Tensor]):
    for l in signals_dict:
        signals_dict[l] = signals_dict[l].detach().cpu()
    return signals_dict
