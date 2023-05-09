
# # fix for the bug: RuntimeError: received 0 items of ancdata
# import resource
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# print(rlimit)
# resource.setrlimit(resource.RLIMIT_NOFILE, (rlimit[1], rlimit[1]))

from experiments.protein_neighborhoods.src.preprocessing.preprocessor_neighborhoods import HDF5Preprocessor
import numpy as np
import itertools
import os
from argparse import ArgumentParser
from progress.bar import Bar
import h5py
import sys
from holographic_vae.so3 import RadialSphericalTensor, MultiChannelRadialSphericalTensor, ZernickeRadialFunctions

from holographic_vae.utils.argparse import *
from experiments.protein_neighborhoods.src.utils import protein_naming
import torch


def callback(nb, selected_masks, selected_weights, frame, rst, mul_rst, rst_normalization):

    try:
        
        # constrain the masks to be only for atoms that we're gonna compute the fourier transform over
        all_selected_masks_mask = np.logical_or.reduce(np.vstack(selected_masks), axis=0)
        selected_masks = [mask[all_selected_masks_mask] for mask in selected_masks]

        all_selected_coeffs = rst.forward_projection_pointwise(torch.tensor(nb['coords'][all_selected_masks_mask])) # assumed to be cartesian coords!
        
        disentangled_coeffs = []
        for mask, weights in zip(selected_masks, selected_weights):
            # integration, with weights, of the fourier transform
            all_selected_coeffs_masked = all_selected_coeffs[mask].float()
            
            if rst_normalization == 'square':
                basesSelfDotsInv = 1.0 / torch.einsum('...f,...f->...', all_selected_coeffs_masked, all_selected_coeffs_masked)
                disentangled_coeffs.append(torch.einsum('...f,...,...->f', all_selected_coeffs_masked, basesSelfDotsInv, torch.tensor(weights).float()))
            elif rst_normalization is None:
                disentangled_coeffs.append(torch.einsum('...f,...->f', all_selected_coeffs_masked, torch.tensor(weights).float()))
            else:
                raise ValueError('Unknown rst_normalization: {}'.format(rst_normalization))
        
        # concatenation of the different channels
        coeffs = mul_rst.combine_inplace(torch.cat(disentangled_coeffs, dim=0))

        if torch.any(torch.isnan(coeffs)):
            print('NaNs in coeffs')
            raise Exception
    
    except Exception as e:
        print(e, file=sys.stderr)
        print(nb['res_id'], file=sys.stderr)
        print('Failed in callback', file=sys.stderr)
        return (None, None, nb['res_id'])

    if frame is not None:
        frame = torch.stack(tuple(map(lambda x: torch.tensor(x), frame))).numpy()

    return (coeffs.numpy(), frame, nb['res_id'])


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--input_hdf5', type=str, required=True,
                        help='Path to hdf5 file containing collected protein neighborhoods. Must be output to the script `get_neighborhoods_pipeline.py`')
    parser.add_argument('--output_hdf5', type=str, required=True,
                        help='User-defined name of output hdf5 file that will contain the zernikegrams.')

    parser.add_argument('--input_dataset_name', type=str, default='data',
                        help='Name of the dataset within input_hdf5 where the neighborhoods information is to be found. We recommend keeping this set to simply "data".')
    parser.add_argument('--output_dataset_name', type=str, default='data',
                        help='Name of the dataset within output_hdf5 where the zernikegram information will be saved. We recommend keeping this set to simply "data".')
    
    parser.add_argument('--parallelism', type=int, default=1,
                        help='Parallelism count for multiprocessing. Keep this set to 1, as the current version runs **slower** with multiprocessing.')
    
    parser.add_argument('--rmax', type=int, default=20,
                        help='Maximum radial order.')
    
    parser.add_argument('--rcut', type=float, default=10.0,
                        help='Radius of the neighborhoods. Alias of "radius".')
    
    parser.add_argument('--lmax', type=int, default=6,
                        help='Maximum spherical order.')
    
    parser.add_argument('--backbone_only', type=str_to_bool, default=False,
                        help='Whether to keep only backbone atoms. Set it to False for H-(V)AE neighborhoods.')
    parser.add_argument('--request_frame', type=str_to_bool, default=False,
                        help='Whether to request the backbone frame. Unused in our experiments.')
    
    parser.add_argument('--rst_normalization', type=optional_str, default='square', options=[None, 'None', 'square'],
                        help='Normalization type per Dirac-Delta projection.')

    parser.add_argument('--channels', type=comma_sep_str_list, default=['C', 'N', 'O', 'S'],
                        help='Atomic and physicochemical channels to be included in the Zernikegrams. Can be any combination of [C, N, O, S, H, SASA, charge].')
    parser.add_argument('--get_psysicochemical_info_for_hydrogens', type=str_to_bool, default=False,
                        help='Whether to include physicochemical information for hydrogens. Only applies if requesting SASA or charge.')

    args = parser.parse_args()


    nRadialFunctions = args.rmax + 1
    p_val = 1 # parity of l-even irreps (1 = even, -1 = odd); must be 1 for projection onto spherical harmonics basis
    p_arg = 1 # p_val*p_arg = parity of l-odd irreps; doesn't matter for the projection onto spherical harmonics basis, only affects how the representation transforms
              # setting this to 1 since the CGNet operates under such assumption, but it really doesn't matter in the context of this script

    ds = HDF5Preprocessor(args.input_hdf5, args.input_dataset_name)

    OnRadialFunctions = ZernickeRadialFunctions(args.rcut, nRadialFunctions, args.lmax, record_zeros = False)
    rst = RadialSphericalTensor(nRadialFunctions, OnRadialFunctions, args.lmax, p_val, p_arg)
    mul_rst = MultiChannelRadialSphericalTensor(rst, len(args.channels))

    labels = []
    projections = []
    frames = []
    n = 0
    data_ids = []
    i = 0
    t = 0

    dt = np.dtype([
        ('res_id','S6', (6)),
        ('zernikegram', 'f4', (mul_rst.dim)),
        ('frame', 'f4', (3, 3)),
        ('label', 'i4', (1))
    ])

    curr_size = 1000
    with h5py.File(args.output_hdf5, 'w') as f:
        # Initialize dataset
        f.create_dataset(args.output_dataset_name,
                         shape=(curr_size,),
                         maxshape=(None,),
                         dtype=dt)

    with h5py.File(args.output_hdf5, 'r+') as f:
        with Bar('Processing', max = ds.count(), suffix='%(percent).1f%%') as bar:
            for proj, frame, nb_id in ds.execute(callback,
                                                 limit = None,
                                                 channels = args.channels,
                                                 backbone_only = args.backbone_only,
                                                 request_frame = args.request_frame,
                                                 get_psysicochemical_info_for_hydrogens = args.get_psysicochemical_info_for_hydrogens,
                                                 params = {'rst': rst, 'mul_rst': mul_rst, 'rst_normalization': args.rst_normalization},
                                                 parallelism = args.parallelism):
                
                t += 1

                if t % 2500 == 0:
                    print('\n\n Status ', i/t, '\n\n', file=sys.stderr)
                
                if proj is None:
                    print(nb_id,' returned error', file=sys.stderr)
                    i += 1
                    bar.next()
                    continue
                
                if nb_id[0].decode('utf-8') not in protein_naming.one_letter_to_aa:
                    print('Got invalid residue type "{}".'.format(nb_id[0].decode('utf-8')), file=sys.stderr)
                    i += 1
                    bar.next()
                    continue

                if args.request_frame:
                    if frame is None:
                        print(nb_id,' returned None frame when frame was requested')
                        i += 1
                        bar.next()
                        continue
                
                if n == curr_size:
                    curr_size += 1000
                    f[args.output_dataset_name].resize((curr_size,))
                
                # save the frame as zeros if it is None, forr compatibility with the provided data type
                if frame is None:
                    frame = np.zeros((3, 3))
                
                f[args.output_dataset_name][n] = (nb_id, proj, frame, protein_naming.aa_to_ind[protein_naming.one_letter_to_aa[nb_id[0].decode('utf-8')]],)
                
                n += 1
                bar.next()

            # finally, resize dataset to be of needed shape to exactly contain the data and nothing more
            f[args.output_dataset_name].resize((n,))

