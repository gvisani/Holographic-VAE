
'''
Purpose: perform inference on ATOm3D LBA neighborhoods

Do it one PDB at a time. Just iterate over one PDB at a time and run HVAE inference on it. Save residue-level representations per PDB in a... numpy array?
'''

import os, sys

from tqdm import tqdm

import json

import gzip, pickle

import numpy as np
import torch
from e3nn import o3

from atom3d.datasets import LMDBDataset

from holographic_vae.nn import SphericalFourierEncoding
from holographic_vae.models import H_VAE
from holographic_vae.so3.functional import put_dict_on_device, make_vec
from holographic_vae.cg_coefficients import get_w3j_coefficients
from holographic_vae.so3 import ZernickeRadialFunctions, RadialSphericalTensor, MultiChannelRadialSphericalTensor
from holographic_vae.utils.loss_functions import *

from experiments.ATOM3D_LBA.src.utils.data import NeighborhoodsTransform, neighborhoods_collate_fn


from typing import *
    


def hvae_inference_atom3d_lba(experiment_dir: str,
                   path_to_raw_data: str,
                   output_filename: str = 'ATOM3D_latent_space.gz',
                   model_name: str = 'lowest_total_loss_with_final_kl_model',
                   verbose: bool = True,
                   loading_bar: bool = True):
    
    '''
    This currently only makes sense to work for batch_size == 1, which processes one PDB at a time.
    To make it work efficiently with multiple PDBs at a time, one would have to keep track of which PDB each neighborhood belongs to, and then aggregate the results for each PDB.
    This is not hard to implement but it doesn't seem to be worth it for the purposes of this.
    '''

    # get hparams from json
    with open(os.path.join(experiment_dir, 'hparams.json'), 'r') as f:
        hparams = json.load(f)

    # seed the random number generator
    rng = torch.Generator().manual_seed(hparams['seed'])

    # setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if verbose: print('Running on %s.' % (device))


    # construct dataset objects using atom3d dataset and custom transform
    transform = NeighborhoodsTransform(elements=hparams['channels'], nb_radius=hparams['rcut'], remove_H=True, remove_water=True, remove_hetero=True, remove_noncanonical_insertion_codes=True, standardize_nonprotein_elements=False)
    dataset = LMDBDataset(path_to_raw_data, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=neighborhoods_collate_fn, drop_last=False)

    OnRadialFunctions = ZernickeRadialFunctions(hparams['rcut'], hparams['rmax']+1, hparams['lmax'], complex_sph = False)
    rst = RadialSphericalTensor(hparams['rmax']+1, OnRadialFunctions, hparams['lmax'], 1, 1)
    mul_rst = MultiChannelRadialSphericalTensor(rst, len(hparams['channels']))
    data_irreps = o3.Irreps(str(mul_rst))

    if verbose: print('Data Irreps: %s' % (str(data_irreps)))
    sys.stdout.flush()


    # load w3j coefficients
    w3j_matrices = get_w3j_coefficients()
    for key in w3j_matrices:
        # if key[0] <= hparams['net_lmax'] and key[1] <= hparams['net_lmax'] and key[2] <= hparams['net_lmax']:
        if device is not None:
            w3j_matrices[key] = torch.tensor(w3j_matrices[key]).float().to(device)
        else:
            w3j_matrices[key] = torch.tensor(w3j_matrices[key]).float()
        w3j_matrices[key].requires_grad = False
    

    # create projection module
    encoding = SphericalFourierEncoding(hparams['lmax'],
                                        hparams['channels'],
                                        sph_normalization='component',
                                        convert_output_to_dict=True,
                                        radial_bases_obj=ZernickeRadialFunctions,
                                        radial_bases_kwargs={'rcut': hparams['rcut'], 'number_of_basis': hparams['rmax']+1, 'lmax': hparams['lmax'], 'complex_sph': False},
                                        rst_normalization=hparams['rst_normalization'])

    
    # create model and load weights
    model = H_VAE(data_irreps, w3j_matrices, hparams['model_hparams'], device, normalize_input_at_runtime=hparams['normalize_input']).to(device)
    model.load_state_dict(torch.load(os.path.join(experiment_dir, model_name + '.pt'), map_location=torch.device(device)))

    num_params = 0
    for param in model.parameters():
        num_params += torch.flatten(param.data).shape[0]
    if verbose: print('There are %d parameters' % (num_params))
    sys.stdout.flush()

    encoding.eval()
    model.eval()

    pdbs, invariants, learned_frames = [], [], []
    
    if loading_bar:
        loading_bar = tqdm
    else:
        loading_bar = lambda x: x
    
    def flatten_list(l):
        return [item for sublist in l for item in sublist]

    for x_coords, x_element, pdb in loading_bar(dataloader):

        # only with batch_size == 1
        x_coords, x_element, pdb = x_coords[0], x_element[0], pdb[0]

        projections = encoding(x_coords, x_element)
        projections = put_dict_on_device(projections, device)

        (z_mean, _), learned_frame = model.encode(projections)

        pdbs.append(pdb)
        invariants.append(z_mean.detach().cpu().numpy())
        learned_frames.append(learned_frame.detach().cpu().numpy())
    

    output = {
        'pdbs': pdbs,
        'invariants': invariants,
        'learned_frames': learned_frames
    }

    with gzip.open(os.path.join(experiment_dir, output_filename), 'wb') as f:
        pickle.dump(output, f)







