
import os, sys
import gzip
import pickle
import numpy as np
import torch
from e3nn import o3
import argparse
from tqdm import tqdm

sys.path.append(os.path.join(sys.path[0], '..'))
from holographic_vae.utils.argparse import *
from spherical_ft import real_sph_ft, real_sph_ift
from lie_learn.spaces.spherical_quadrature import estimate_spherical_quadrature_weights

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='./data',
                        help='Path to file containing data on the S2 grid.')

    parser.add_argument('--output_path', type=str, default='./data',
                        help='Path to desired output file.')

    parser.add_argument('--grids_path', type=str, default='./data',
                        help='Path to file containing S2 grid onto which signals are defined.')

    parser.add_argument('--input_type', type=str, required=True,
                        help='Rotation type of data, in {train/val}{test} format. NRNR, RR, NRR or RNR.')             

    parser.add_argument('--bw', type=int, default=30,
                        help='Bandwidth of the grid to be requested')

    parser.add_argument('--lmax', type=int, default=10,
                        help='Maximum value of l for which to compute spherical harmonics projections.')

    parser.add_argument('--cz', type=int, default=10,
                        help='Chunk size of digits to which the same rotation was applied. Irrelevant for input_type==NRNR, but must match the value used in `gendata.py`.')

    parser.add_argument('--normalize', type=optional_str, default='avg_sqrt_power', choices=[None, 'None', 'avg_sqrt_power'],
                        help='Per-datapoint normalization on the forward fourier transform.')

    parser.add_argument('--complex_sph', type=str_to_bool, default=False,
                        help='Whether to project signals onto the complex spherical harmonics basis.')

    parser.add_argument('--quad_weights', type=str_to_bool, default=True,
                        help='Whether to use quad_weights computed with lie_learn to compute the forward FT.')

    args = parser.parse_args()

    input_type_map = {
        'NRNR': '-no_rotate_train-no_rotate_test-cz=%d' % (args.cz),
        'RR': '-cz=%d' % (args.cz),
        'NRR': '-no_rotate_train-cz=%d' % (args.cz),
        'RNR': '-no_rotate_test-cz=%d' % (args.cz)
    }
    input_file = os.path.join(args.input_path, 's2_mnist%s-b=%d.gz' % (input_type_map[args.input_type], args.bw))
    output_file = os.path.join(args.output_path, 'real_sph_mnist%s-b=%d-lmax=%d-normalize=%s-quad_weights=%s.gz' % (input_type_map[args.input_type], args.bw, args.lmax, args.normalize, args.quad_weights))

    
    with gzip.open(input_file, 'rb') as f:
        dataset = pickle.load(f)
    
    with gzip.open(os.path.join(args.grids_path, 'ba_grid-b=%d.gz' % (args.bw)), 'rb') as f:
        grid = pickle.load(f)

        # flatten grid elements
        grid = list(grid)
        for i in range(2):
            grid[i] = grid[i].flatten()
        grid = tuple(grid)
    
    for label in ['train', 'valid', 'test']:
        N = dataset[label]['images'].shape[0]
        dataset[label]['images'] = dataset[label]['images'].reshape((N, -1))

    ## Pre-compute quadrature weights for the grid
    # We need to compute the intergal of the forward ft as accurately as possible in order to be able to
    #   compute the inverse ft and reconstruct the original signals.
    # The package 'lie_learn' estimates the quadrature weights of given points via least squares regression.
    #   They implement their own real spherical harmonics but it looks like they appear to me to be pretty much equivalent.
    # In the folder 'quadrature_figures' one can see that not using quadrature weights makes it way harder
    #   to reconstruct the original signals.
    # Ideally, one would probably construct the grid and the weights at the same time with the needed symmetries in order
    #   to get the intergal as exact as possible (instead of using a least squares approximation of the weights). That would
    #   probably generate the most accurate ft projections and resulting reconstructions.
    if args.quad_weights:
        try:
            quad_weights = np.load(os.path.join(args.grids_path, 'quad_weights-b=%d-lmax=%d.npy' % (args.bw, args.lmax)))
        except:
            print('Precomputed quad weights not found. Computing them.')
            quad_weights, _, _, _ = estimate_spherical_quadrature_weights(np.transpose(np.vstack(grid)), args.lmax, normalization='seismology', condon_shortley=True)
            np.save(os.path.join(args.grids_path, 'quad_weights-b=%d-lmax=%d.npy' % (args.bw, args.lmax)), quad_weights)
    else:
        quad_weights = None

    projections_dataset = {}
    train_norm_factors = []
    for split in ['train', 'valid', 'test']:
        projections_dataset[split] = {}
        projections_dataset[split]['rotations'] = dataset[split]['rotations']
        projections_dataset[split]['labels'] = dataset[split]['labels']

        all_signals = dataset[split]['images']
        if args.complex_sph:
            raise NotImplementedError('Projection on complex SPH basis not yet implemented.')
        else:
            # projections_dataset[split]['projections'] = real_sph_ft(signals, grid, args.lmax, quad_weights)
            batch_size = 1000
            num_batches = all_signals.shape[0] // batch_size
            projections = []
            for i in tqdm(range(num_batches)):
                signals = all_signals[i*batch_size : (i+1)*batch_size]

                proj, norm_factors = real_sph_ft(signals, grid, args.lmax, quad_weights, args.normalize)

                if split == 'train':
                    train_norm_factors.append(norm_factors)

                projections.append(proj)
            
            if split == 'train':
                train_norm_factors = torch.mean(torch.cat(train_norm_factors, dim=-1))
            
            projections_dataset[split]['projections'] = torch.vstack(projections) / train_norm_factors

    with gzip.open(output_file, 'wb') as f:
        pickle.dump(projections_dataset, f)


    
