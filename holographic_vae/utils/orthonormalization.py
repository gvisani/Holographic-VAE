
import torch
from torch import Tensor

def orthonormalize_frame(x_psy_N6: Tensor):
    '''
    Gram-Schmidt process
    
    y = psy - (<x, psy> / <x, x>) x
    z = x \cross y
    '''
    
    x, psy = x_psy_N6[:, :3], x_psy_N6[:, 3:]
    
    x_dot_psy = torch.sum(torch.mul(x, psy), dim=1).view(-1, 1)
    x_dot_x = torch.sum(torch.mul(x, x), dim=1).view(-1, 1)
    
    y = psy - (x_dot_psy/x_dot_x) * x
    
    z = torch.cross(x, y, dim=1)
    
    x = x / torch.sqrt(torch.sum(torch.mul(x, x), dim=1).view(-1, 1))
    y = y / torch.sqrt(torch.sum(torch.mul(y, y), dim=1).view(-1, 1))
    z = z / torch.sqrt(torch.sum(torch.mul(z, z), dim=1).view(-1, 1))
    
    xyz = torch.cat([x, y, z], dim=1)
    
    return xyz