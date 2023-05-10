
import numpy as np
import torch
import e3nn
from e3nn import o3

from lie_learn.spaces.spherical_quadrature import estimate_spherical_quadrature_weights


def real_sph_ft(signals_BN, grid_N, lmax, quad_weights_N=None, normalization=None):
    '''
    Note: quadrature weights are expensive to compute, so precompute them in advance and provide them to this function
        - `quad_weights, residuals, rank, s = estimate_spherical_quadrature_weights(np.transpose(np.vstack(grid_N)), lmax, normalization='seismology', condon_shortley=True)`
        - We need to compute the intergal of the forward ft as accurately as possible in order to be able to
          compute the inverse ft and reconstruct the original signals.
          The package 'lie_learn' estimates the quadrature weights of given points via least squares regression.
          They implement their own real spherical harmonics but it looks like they appear to me to be pretty much equivalent.
          In the folder 'quadrature_figures' one can see that not using quadrature weights makes it way harder
          to reconstruct the original signals.
          Ideally, one would probably construct the grid and the weights at the same time with the needed symmetries in order
          to get the intergal as exact as possible (instead of using a least squares approximation of the weights). That would
          probably generate the most accurate ft projections and resulting reconstructions.
    '''

    # Set quadrature weights to one if they are not specified
    if quad_weights_N is None:
        quad_weights_N = np.ones(grid_N[0].shape)

    (beta_N, alpha_N) = grid_N
    beta_N = torch.tensor(beta_N)
    alpha_N = torch.tensor(alpha_N)

    sph_NF = o3.spherical_harmonics_alpha_beta(range(lmax+1), alpha_N, beta_N)
    signals_BN = torch.tensor(signals_BN)
    quad_weights_N = torch.tensor(quad_weights_N)
    projections_BF = torch.einsum('nf,bn,n->bf', sph_NF, signals_BN, quad_weights_N).float()

    norm_factors = None

    if normalization == 'square':
        irreps = o3.Irreps.spherical_harmonics(lmax, 1)
        projSelfDotInv_B = 1.0 / torch.einsum('bf,bf->b', projections_BF, projections_BF)
        projections_BF = torch.einsum('bf,b->bf', projections_BF, projSelfDotInv_B)

    elif normalization == 'power':
        irreps = o3.Irreps.spherical_harmonics(lmax, 1)
        ls_indices = torch.cat([torch.tensor(irreps.ls)[torch.tensor(irreps.ls) == l].repeat(2*l+1) for l in sorted(list(set(irreps.ls)))]).type(torch.float)
        projSelfDotInv_B = 1.0 / torch.einsum('bf,bf,f->b', projections_BF, projections_BF, 1.0 / (2*ls_indices + 1))
        projections_BF = torch.einsum('bf,b->bf', projections_BF, projSelfDotInv_B)

    elif normalization == 'sqrt_power':
        irreps = o3.Irreps.spherical_harmonics(lmax, 1)
        ls_indices = torch.cat([torch.tensor(irreps.ls)[torch.tensor(irreps.ls) == l].repeat(2*l+1) for l in sorted(list(set(irreps.ls)))]).type(torch.float)
        projSelfDotInv_B = 1.0 / torch.sqrt(torch.einsum('bf,bf,f->b', projections_BF, projections_BF, 1.0 / (2*ls_indices + 1)))
        projections_BF = torch.einsum('bf,b->bf', projections_BF, projSelfDotInv_B)

    elif normalization == 'avg_sqrt_power':
        irreps = o3.Irreps.spherical_harmonics(lmax, 1)
        ls_indices = torch.cat([torch.tensor(irreps.ls)[torch.tensor(irreps.ls) == l].repeat(2*l+1) for l in sorted(list(set(irreps.ls)))]).type(torch.float)
        norm_factors = torch.sqrt(torch.einsum('bf,bf,f->b', projections_BF, projections_BF, 1.0 / (2*ls_indices + 1)))
    
    elif normalization == 'magnitudes':
        irreps = o3.Irreps.spherical_harmonics(lmax, 1)
        ls_indices = torch.cat([torch.tensor(irreps.ls)[torch.tensor(irreps.ls) == l].repeat(2*l+1) for l in sorted(list(set(irreps.ls)))]).type(torch.float)
        projSelfDot_BF = torch.einsum('bf,bf,f->bf', projections_BF, projections_BF, 1.0 / (2*ls_indices + 1))
        ls_indices_int = ls_indices.type(torch.int)
        projSelfDot_BF_new = []
        for l in sorted(list(set(list(ls_indices_int.numpy())))):
            projSelfDot_BF_new.append(torch.Tensor([torch.sum(projSelfDot_BF[:, ls_indices_int == l], dim=1)]).repeat(1, 2*l+1))
        projSelfDot_BF_new = torch.sqrt(torch.cat(projSelfDot_BF_new, dim=-1))
        projections_BF = torch.einsum('bf,bf->bf', projections_BF, 1.0 / projSelfDot_BF_new.float())

    elif normalization is None or normalization == 'None':
        pass

    else:
        raise Exception
    
    return projections_BF, norm_factors

def real_sph_ift(projections_BF, grid_N, lmax):
    (beta_N, alpha_N) = grid_N
    beta_N = torch.tensor(beta_N)
    alpha_N = torch.tensor(alpha_N)
    sph_NF = o3.spherical_harmonics_alpha_beta(range(lmax+1), alpha_N, beta_N)
    rec_signal_BN = torch.einsum('bf,nf->bn', projections_BF.float(), sph_NF.float())
    return rec_signal_BN


def complex_sph_ft():
    pass

def complex_sph_ift():
    pass