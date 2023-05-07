
import os, sys
import numpy as np
import torch

from holographic_vae.utils.conversions import spherical_to_cartesian__numpy
from experiments.protein_neighborhoods.src.utils.protein_naming import aa_to_ind_size, one_letter_to_aa
from holographic_vae.so3 import ZernickeRadialFunctions, RadialSphericalTensor, MultiChannelRadialSphericalTensor
from typing import *
from torch import Tensor

BACKBONE_ATOMS = [b' N  ', b' CA ', b' C  ', b' O  ']
BACKBONE_ATOMS_PLUS_CB = [b' N  ', b' CA ', b' C  ', b' O  ', b' CB ']

def get_zernikegrams(nbs: np.ndarray, # of custom dtype
                        rcut: float,
                        rmax: int,
                        lmax: int,
                        channels: List[str], 
                        backbone_only: bool = False,
                        request_frame: bool = False,
                        get_psysicochemical_info_for_hydrogens: bool = True,
                        rst_normalization: Optional[str] = None
                       ) -> dict:


    OnRadialFunctions = ZernickeRadialFunctions(rcut, rmax+1, lmax, complex_sph = False)
    rst = RadialSphericalTensor(rmax+1, OnRadialFunctions, lmax, 1, 1)
    mul_rst = MultiChannelRadialSphericalTensor(rst, len(channels))
    
    zernikegrams = []
    data_ids = []
    frames = []
    labels = []
    for nb in nbs:

        selected_masks, selected_weights, frame = extract_neighborhood_info(nb,
                                                                            channels=channels,
                                                                            backbone_only=backbone_only,
                                                                            request_frame=request_frame,
                                                                            get_psysicochemical_info_for_hydrogens=get_psysicochemical_info_for_hydrogens)
        
        zgram = get_zernikegram_of_neighborhood(nb, selected_masks, selected_weights, rst, mul_rst, rst_normalization=rst_normalization)

        if frame is None and request_frame:
            print('Error: frame is None when requested. Skipping neighborhood.', file=sys.stderr)
            continue
        elif frame is not None:
            frames.append(frame)

        zernikegrams.append(zgram)
        data_ids.append(nb['res_id'])
        labels.append(aa_to_ind_size[one_letter_to_aa[nb['res_id'][0].decode('utf-8')]])

    zernikegrams = np.vstack(zernikegrams)

    data_ids = np.array(data_ids)
    if request_frame:
        frames = np.vstack(frames).reshape(-1, 3, 3)
    else:
        frames = None
    labels = np.hstack(labels)

    print(np.count_nonzero(np.isinf(zernikegrams)))

    return {'zernikegrams': zernikegrams,
            'data_ids': data_ids,
            'frames': frames,
            'labels': labels}


def get_zernikegram_of_neighborhood(nb: np.ndarray,
                                    selected_masks: np.ndarray,
                                    selected_weights: np.ndarray,
                                    rst: RadialSphericalTensor,
                                    mul_rst: MultiChannelRadialSphericalTensor,
                                    rst_normalization: Optional[str] = None
                                    ):

    # constrain the masks to be only for atoms that we're gonna compute the fourier transform over
    all_selected_masks_mask = np.logical_or.reduce(np.vstack(selected_masks), axis=0)
    selected_masks = [mask[all_selected_masks_mask] for mask in selected_masks]

    all_selected_coeffs = rst.forward_projection_pointwise(torch.tensor(nb['coords'][all_selected_masks_mask])) # assumed to be cartesian coords!
    
    disentangled_coeffs = []
    for mask, weights in zip(selected_masks, selected_weights):
        # integration, with weights, of the fourier transform
        if rst_normalization == 'square':
            basesSelfDotsInv = 1.0 / torch.einsum('...f,...f->...', all_selected_coeffs[mask], all_selected_coeffs[mask])
            coeffs = torch.einsum('...f,...,...->f', all_selected_coeffs[mask], basesSelfDotsInv, torch.tensor(weights))
        elif rst_normalization is None:
            disentangled_coeffs.append(torch.einsum('...f,...->f', all_selected_coeffs[mask], torch.tensor(weights)))
        else:
            raise ValueError('Unknown rst_normalization: {}'.format(rst_normalization))
    
    # concatenation of the different channels
    coeffs = mul_rst.combine_inplace(torch.cat(disentangled_coeffs, dim=0))
    
    return coeffs.numpy()

def get_zernikegram_of_neighborhood_maybe_faster(nb: np.ndarray,
                                    selected_masks: np.ndarray,
                                    selected_weights: np.ndarray,
                                    rst: RadialSphericalTensor,
                                    mul_rst: MultiChannelRadialSphericalTensor
                                    ):

    # constrain the masks to be only for atoms that we're gonna compute the fourier transform over
    all_selected_masks_mask = np.logical_or.reduce(np.vstack(selected_masks), axis=0)
    selected_masks = [mask[all_selected_masks_mask] for mask in selected_masks]

    all_selected_coeffs = rst.forward_projection_pointwise(torch.tensor(nb['coords'][all_selected_masks_mask])) # assumed to be cartesian coords!
    
    disentangled_coeffs = []
    for mask, weights in zip(selected_masks, selected_weights):
        # integration, with weights, of the fourier transform
        disentangled_coeffs.append(torch.einsum('...f,...->f', all_selected_coeffs[mask], torch.tensor(weights)))
    
    # concatenation of the different channels
    coeffs = mul_rst.combine_inplace(torch.cat(disentangled_coeffs, dim=0))
    
    return coeffs.numpy()


def extract_neighborhood_info(nb,
                              channels=[],
                              backbone_only=False,
                              request_frame=False,
                              get_psysicochemical_info_for_hydrogens=True):

    '''
    Collects boolean masks indicating which points belong to a specific channel. Also collects weights for those specific points.
    Note that `selected_mask.shape = (max_atoms,)` and `selected_weights.shape = (np.sum(selected_mask),)`, where 

    Assume physicochemical info is provided AFTER all the atom types

    get_psysicochemical_info_for_hydrogens only applies when backbone_only is False

    return all coordinates of interest, along with selecting_indices and values for compuing individual point clouds
    '''

    try:

        cartesian_coords = nb['coords'] # already in cartesian coordinates

        if backbone_only:
            if len(nb['atom_names'][0]) != 4:
                print('Bug!: ', nb['atom_names'][0], file=sys.stderr)
                raise Exception
            
            selected_masks = []
            selected_weights = []
            for channel in channels:
                if channel == 'CA':
                    selected_masks.append(nb['atom_names'] == b' CA ')
                    selected_weights.append(np.ones(np.sum(selected_masks[-1])))
                elif channel == 'C':
                    selected_masks.append(nb['atom_names'] == b' C  ')
                    selected_weights.append(np.ones(np.sum(selected_masks[-1])))
                elif channel == 'N':
                    selected_masks.append(nb['atom_names'] == b' N  ')
                    selected_weights.append(np.ones(np.sum(selected_masks[-1])))
                elif channel == 'O':
                    selected_masks.append(nb['atom_names'] == b' O  ')
                    selected_weights.append(np.ones(np.sum(selected_masks[-1])))
                # elif channel == 'charge':
                #     selected_indices.append(np.logical_or.reduce(np.vstack(selected_masks), axis=0))
                #     selected_weights.append(nb['charge'][selected_masks[-1]])
                else:
                    raise Exception('Unknown channel: {}'.format(channel))

        else:
            
            selected_masks = []
            selected_weights = []
            for channel in channels:
                if channel == 'C':
                    selected_masks.append(nb['elements'] == b'C')
                    selected_weights.append(np.ones(np.sum(selected_masks[-1])))
                elif channel == 'N':
                    selected_masks.append(nb['elements'] == b'N')
                    selected_weights.append(np.ones(np.sum(selected_masks[-1])))
                elif channel == 'O':
                    selected_masks.append(nb['elements'] == b'O')
                    selected_weights.append(np.ones(np.sum(selected_masks[-1])))
                elif channel == 'S':
                    selected_masks.append(nb['elements'] == b'S')
                    selected_weights.append(np.ones(np.sum(selected_masks[-1])))
                elif channel == 'H':
                    selected_masks.append(nb['elements'] == b'H')
                    selected_weights.append(np.ones(np.sum(selected_masks[-1])))
                elif channel == 'SASA':
                    if not get_psysicochemical_info_for_hydrogens:
                        selected_masks.append(np.logical_and(nb['elements'] != b'', nb['elements'] != b'H'))
                    else:
                        selected_masks.append(nb['elements'] != b'')
                    selected_weights.append(nb['SASAs'][selected_masks[-1]])
                elif channel == 'charge':
                    if not get_psysicochemical_info_for_hydrogens:
                        selected_masks.append(np.logical_and(nb['elements'] != b'', nb['elements'] != b'H'))
                    else:
                        selected_masks.append(nb['elements'] != b'')
                    selected_weights.append(nb['charges'][selected_masks[-1]])
                else:
                    raise Exception('Unknown channel: {}'.format(channel))

        if request_frame:
            try:
                central_res = np.logical_and.reduce(nb['res_ids'] == nb['res_id'], axis=-1)
                central_CA_coords = np.array([0.0, 0.0, 0.0]) # since we centered the neighborhood on the alpha carbon
                central_N_coords = np.squeeze(cartesian_coords[central_res][nb['atom_names'][central_res] == b' N  '])
                central_C_coords = np.squeeze(cartesian_coords[central_res][nb['atom_names'][central_res] == b' C  '])

                # if central_N_coords.shape[0] == 3:
                #     print('-----'*16)
                #     print(nb['res_id'])
                #     print(nb['res_ids'])
                #     print(nb['atom_names'])
                #     print(central_N_coords)
                #     print(central_C_coords)
                #     print(nb['atom_names'].shape)
                #     print(nb['coords'].shape)
                #     print('-----'*16)

                # assert that there is only one atom with three coordinates
                assert (central_CA_coords.shape[0] == 3), 'first assert'
                assert (len(central_CA_coords.shape) == 1), 'second assert'
                assert (central_N_coords.shape[0] == 3), 'third assert'
                assert (len(central_N_coords.shape) == 1), 'fourth assert'
                assert (central_C_coords.shape[0] == 3), 'fifth assert'
                assert (len(central_C_coords.shape) == 1), 'sixth assert'

                # y is unit vector perpendicular to x and lying on the plane between CA_N (x) and CA_C
                # z is unit vector perpendicular to x and the plane between CA_N (x) and CA_C
                x = central_N_coords - central_CA_coords
                x = x / np.linalg.norm(x)

                CA_C_vec = central_C_coords - central_CA_coords

                z = np.cross(x, CA_C_vec)
                z = z / np.linalg.norm(z)

                y = np.cross(z, x)

                frame = (x, y, z)
                    
            except Exception as e:
                print(e)
                print('No central residue (or other unwanted error).')
                frame = None
        else:
            frame = None

    except Exception as e:
        print(e)
        print(nb['res_id'])
        print('Failed in process_data')
        return False
    
    return selected_masks, selected_weights, frame