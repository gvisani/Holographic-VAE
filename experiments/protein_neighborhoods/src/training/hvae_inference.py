
import os, sys
import gzip, pickle
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pylab
import pandas as pd
from holographic_vae.so3.radial_spherical_tensor import MultiChannelRadialSphericalTensor

import torch
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
import e3nn
from e3nn import o3
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm

import umap

from typing import *

from holographic_vae.models import H_VAE
from holographic_vae.so3.functional import put_dict_on_device, make_vec
from holographic_vae.cg_coefficients import get_w3j_coefficients
from holographic_vae.so3 import ZernickeRadialFunctions, RadialSphericalTensor, MultiChannelRadialSphericalTensor
from holographic_vae.utils.loss_functions import *

from experiments.protein_neighborhoods.src.utils.data import NeighborhoodsDataset
from experiments.protein_neighborhoods.src.preprocessing import get_neighborhoods, get_structural_info, get_zernikegrams
from experiments.protein_neighborhoods.src.utils import protein_naming


'''
NOTE: ensure the new code on processing .pdb file works as expected
'''


def hvae_inference(experiment_dir: str,
                    output_filepath: Optional[str] = None,
                    data_filepath: Optional[str] = None,
                    pdb_dir: Optional[str] = None,
                    input_dataset_name: Optional[str] = None, # only relevant is data_filepath is an hdf5 file
                    normalize_input_at_runtime: bool = True,
                    model_name: str = 'lowest_total_loss_with_final_kl_model',
                    n_finetuning_epochs: int = 0,
                    verbose: bool = False,
                    loading_bar: bool = True,
                    batch_size: int = 100):

    # get hparams from json
    with open(os.path.join(experiment_dir, 'hparams.json'), 'r') as f:
        hparams = json.load(f)

    # seed the random number generator
    rng = torch.Generator().manual_seed(hparams['seed'])

    # setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if verbose: print('Running on %s.' % (device))


    ########## THE CODE BLOCK BELOW MAY BE CHANGED TO ACCOMODATE A DIFFERENT DATA-LOADING PIPELINE ##########

    if data_filepath is None: # run on default test data
        from experiments.protein_neighborhoods.src.utils.data import load_data
        output_filepath = os.path.join(experiment_dir, 'test_data_results-{}.npz'.format(model_name))
        datasets, data_irreps, _ = load_data(hparams, splits=['test'])
        test_dataset = datasets['test']
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, generator=rng, shuffle=False, drop_last=False)

    elif data_filepath[-4:] == '.pdb' or (data_filepath[-4:] == '.csv' and pdb_dir is not None):

        from holographic_vae.so3.functional import filter_by_l

        if data_filepath[-4:] == '.pdb': # --> single pdb is provided
            protein = get_structural_info(data_filepath)
            nbs = get_neighborhoods(protein, remove_central_residue = False, backbone_only = False)

        else: # (data_filepath[-4:] == '.csv' and pdb_dir is not None) --> list of pdbs is provided
            pdb_list = pd.read_csv(data_filepath)['pdb'].tolist()
            if verbose: print('Collecting neighborhoods from %d PDB files...' % len(pdb_list))
            sys.stdout.flush()
            
            proteins = get_structural_info([os.path.join(pdb_dir, pdb+'.pdb') for pdb in pdb_list])
            nbs = get_neighborhoods(proteins, remove_central_residue = False, backbone_only = False)

        orig_OnRadialFunctions = ZernickeRadialFunctions(hparams['rcut'], hparams['rmax']+1, hparams['collected_lmax'], complex_sph = False)
        orig_rst = RadialSphericalTensor(hparams['rmax']+1, orig_OnRadialFunctions, hparams['collected_lmax'], 1, 1)
        orig_mul_rst = MultiChannelRadialSphericalTensor(orig_rst, len(hparams['channels']))
        orig_data_irreps = o3.Irreps(str(orig_mul_rst))

        if verbose: print('Generating zernikegrams...')
        sys.stdout.flush()
        zgrams_data = get_zernikegrams(nbs, hparams['rcut'], hparams['rmax'], hparams['collected_lmax'], hparams['channels'], backbone_only=hparams['backbone_only'], request_frame=False, get_psysicochemical_info_for_hydrogens=hparams['get_psysicochemical_info_for_hydrogens'], rst_normalization=hparams['rst_normalization'])

        if hparams['lmax'] != hparams['collected_lmax']: # this slicing is only relevant when using rst_normalization == 'square'. We recommend keeping 'collected_lmax' and 'lmax' the same.
            zgrams_data['zernikegrams'] = filter_by_l(torch.tensor(zgrams_data['zernikegrams']), orig_data_irreps, hparams['lmax']).float()
        else:
            zgrams_data['zernikegrams'] = torch.tensor(zgrams_data['zernikegrams']).float()
        
        # requested data irreps
        # note: no need to filter by channel since the zernikegrams are being computed on-the-fly with the requsted channels, instead of having been pre-collected.
        OnRadialFunctions = ZernickeRadialFunctions(hparams['rcut'], hparams['rmax']+1, hparams['lmax'], complex_sph = False)
        rst = RadialSphericalTensor(hparams['rmax']+1, OnRadialFunctions, hparams['lmax'], 1, 1)
        mul_rst = MultiChannelRadialSphericalTensor(rst, len(hparams['channels']))
        data_irreps = o3.Irreps(str(mul_rst))
        ls_indices = torch.cat([torch.tensor(data_irreps.ls)[torch.tensor(data_irreps.ls) == l].repeat(2*l+1) for l in sorted(list(set(data_irreps.ls)))]).type(torch.float)
        
        if hparams['normalize_input']:
            normalize_input_at_runtime = True
        else:
            normalize_input_at_runtime = False
        
        def stringify(data_id):
            return '_'.join(list(map(lambda x: x.decode('utf-8'), list(data_id))))
    
        if zgrams_data['frames'] is None:
            zgrams_data['frames'] = np.zeros((zgrams_data['labels'].shape[0], 3, 3))

        if verbose: print('Power: %.4f' % (torch.mean(torch.sqrt(torch.einsum('bf,bf,f->b', zgrams_data['zernikegrams'][:1000], zgrams_data['zernikegrams'][:1000], 1.0 / (2*ls_indices + 1)))).item()))

        test_dataset = NeighborhoodsDataset(zgrams_data['zernikegrams'], data_irreps, torch.tensor(zgrams_data['labels']), list(zip(list(zgrams_data['frames']), list(map(stringify, zgrams_data['data_ids'])))))
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, generator=rng, shuffle=False, drop_last=False)

    elif data_filepath[-5:] == '.hdf5': # inference with hdf5 file using standard naming of variables

        import h5py
        from holographic_vae.so3.functional import filter_by_l
        from holographic_vae.so3.functional import filter_by_channels


        if 'n_channels' in hparams: # signals old way of loading data and generating zernikegrams
            # BEWARE: this may change, currently for LBA
            orig_OnRadialFunctions = ZernickeRadialFunctions(hparams['rcut'], hparams['rmax']+1, 8, complex_sph = False)
            orig_rst = RadialSphericalTensor(hparams['rmax']+1, orig_OnRadialFunctions, 8, 1, 1)
            orig_mul_rst = MultiChannelRadialSphericalTensor(orig_rst, 7)
            orig_data_irreps = o3.Irreps(str(orig_mul_rst))

            OnRadialFunctions = ZernickeRadialFunctions(hparams['rcut'], hparams['rmax']+1, hparams['lmax'], complex_sph = False)
            rst = RadialSphericalTensor(hparams['rmax']+1, OnRadialFunctions, hparams['lmax'], 1, 1)
            mul_rst = MultiChannelRadialSphericalTensor(rst, hparams['n_channels'])
            data_irreps = o3.Irreps(str(mul_rst))
            n_channels = hparams['n_channels']
        else:
            # BEWARE: this may change, currently for LBA
            orig_OnRadialFunctions = ZernickeRadialFunctions(hparams['rcut'], hparams['rmax']+1, 6, complex_sph = False)
            orig_rst = RadialSphericalTensor(hparams['rmax']+1, orig_OnRadialFunctions, 6, 1, 1)
            orig_mul_rst = MultiChannelRadialSphericalTensor(orig_rst, 4)
            orig_data_irreps = o3.Irreps(str(orig_mul_rst))
            n_channels = len(hparams['channels'])
        
        def stringify(data_id):
            return '_'.join(list(map(lambda x: x.decode('utf-8'), list(data_id))))
        
        with h5py.File(data_filepath, 'r') as f:
            mask = ~np.logical_and.reduce(f[input_dataset_name]['res_id'] == np.array([b'', b'', b'', b'', b'', b'']), axis=1) # select non-empty zernikegrams only

            test_data = torch.tensor(f[input_dataset_name]['zernikegram'][mask]).float()
            if orig_rst.lmax != hparams['lmax']:
                print('Filtering \ell...')
                test_data = filter_by_l(test_data, orig_data_irreps, hparams['lmax']).float()

            new_l_OnRadialFunctions = ZernickeRadialFunctions(hparams['rcut'], hparams['rmax']+1, hparams['lmax'], complex_sph = False)
            new_l_rst = RadialSphericalTensor(hparams['rmax']+1, new_l_OnRadialFunctions, hparams['lmax'], 1, 1)
            new_l_mul_rst = MultiChannelRadialSphericalTensor(new_l_rst, orig_mul_rst.num_channels)

            print('Original number of channels: %d' % orig_mul_rst.num_channels)
            print('New \ell number of channels: %d' % new_l_mul_rst.num_channels)

            if orig_mul_rst.num_channels != len(hparams['channels']):
                print('Filtering channels...')
                test_data = filter_by_channels(test_data, new_l_mul_rst, np.arange(n_channels))

            OnRadialFunctions = ZernickeRadialFunctions(hparams['rcut'], hparams['rmax']+1, hparams['lmax'], complex_sph = False)
            rst = RadialSphericalTensor(hparams['rmax']+1, OnRadialFunctions, hparams['lmax'], 1, 1)
            mul_rst = MultiChannelRadialSphericalTensor(rst, len(hparams['channels']))
            data_irreps = o3.Irreps(str(mul_rst))

            print('New number of channels: %d' % mul_rst.num_channels)
            
            sys.stdout.flush()

            ls_indices = torch.cat([torch.tensor(data_irreps.ls)[torch.tensor(data_irreps.ls) == l].repeat(2*l+1) for l in sorted(list(set(data_irreps.ls)))]).type(torch.float)
            print('Power: %.4f' % (torch.mean(torch.sqrt(torch.einsum('bf,bf,f->b', test_data[:1000], test_data[:1000], 1.0 / (2*ls_indices + 1)))).item()))

            test_labels = torch.tensor(f[input_dataset_name]['label'][mask])
            test_ids = np.array(list(map(stringify, f[input_dataset_name]['res_id'][mask])))

            try:
                test_frames = torch.tensor(f[input_dataset_name]['frame'][mask])
            except Exception as e:
                print('Warning: no frames.', file=sys.stderr)
                print(e)
                test_frames = np.zeros((test_labels.shape[0], 3, 3))

        if hparams['normalize_input']:
            normalize_input_at_runtime = True
        else:
            normalize_input_at_runtime = False
        
        test_dataset = NeighborhoodsDataset(torch.tensor(test_data), data_irreps, torch.tensor(test_labels), list(zip(list(test_frames), list(test_ids))))
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, generator=rng, shuffle=False, drop_last=False)

    else:
        raise NotImplementedError()
    
    if verbose: print('Done preprocessing.')
    sys.stdout.flush()
    
    ########## THIS CODE BLOCK ABOVE MAY BE CHANGED TO ACCOMODATE A DIFFERENT DATA-LOADING PIPELINE ##########

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
    
    # create model and load weights
    model = H_VAE(data_irreps, w3j_matrices, hparams['model_hparams'], device, normalize_input_at_runtime=normalize_input_at_runtime).to(device)
    model.load_state_dict(torch.load(os.path.join(experiment_dir, model_name + '.pt'), map_location=torch.device(device)))

    num_params = 0
    for param in model.parameters():
        num_params += torch.flatten(param.data).shape[0]
    if verbose: print('There are %d parameters' % (num_params))
    sys.stdout.flush()

    if n_finetuning_epochs > 0:
        if verbose: print('Fine-tuning model for %d epochs' % (n_finetuning_epochs))
        model = finetune(model, hparams, n_finetuning_epochs, test_dataloader, device)

    model.eval()
    invariants, labels, learned_frames, images, rec_images, data_ids = [], [], [], [], [], []
    
    if loading_bar:
        loadind_bar = tqdm
    else:
        loadind_bar = lambda x: x

    for i, (X, X_vec, y, (rot, data_id)) in loadind_bar(enumerate(test_dataloader)):
        X = put_dict_on_device(X, device)
        rot = rot.view(-1, 3, 3).float().to(device)

        (z_mean, _), learned_frame = model.encode(X)

        z = z_mean

        if hparams['model_hparams']['learn_frame']:
            x_reconst = model.decode(z, learned_frame)
        else:
            x_reconst = model.decode(z, rot)

        invariants.append(z.detach().cpu().numpy())

        if hparams['model_hparams']['learn_frame']:
            learned_frames.append(learned_frame.reshape(-1, 1, 9).squeeze(1).detach().cpu().numpy())
        else:
            learned_frames.append(rot.reshape(-1, 1, 9).squeeze(1).cpu().numpy())
        
        labels.append(y.cpu().numpy())
        images.append(X_vec.detach().cpu().numpy())
        rec_images.append(make_vec(x_reconst).detach().cpu().numpy())
        data_ids.append(data_id)
        
    invariants_ND = np.vstack(invariants)
    learned_frames_N9 = np.vstack(learned_frames)
    labels_N = np.hstack(labels)
    data_ids_N = np.hstack(data_ids) # hstack because we assume the data_ids are represented in their stringified version. otherwise we would have to use vstack
    images_NF = np.vstack(images)
    rec_images_NF = np.vstack(rec_images)
    
    cosine_loss_fn = eval(NAME_TO_LOSS_FN['cosine'])(data_irreps, device)
    cosine_loss = cosine_loss_fn(torch.tensor(images_NF).float().to(device), torch.tensor(rec_images_NF).float().to(device)).item()
    mse_loss = torch.nn.functional.mse_loss(torch.tensor(images_NF).float().to(device), torch.tensor(rec_images_NF).float().to(device)).item()
    if verbose: print('Cosine loss: {:.3f}'.format(cosine_loss))
    # if verbose: print('MSE loss: {:.8f}'.format(mse_loss))

    if verbose: print('Done running model, saving to %s ...' % (output_filepath))
    sys.stdout.flush()

    if output_filepath is None: # then return everything
        return invariants_ND, learned_frames_N9, labels_N, data_ids_N, images_NF, rec_images_NF, cosine_loss
    
    elif output_filepath[-4:] == '.npz':
        np.savez(output_filepath,
                        invariants_ND = invariants_ND,
                        learned_frames_N9 = learned_frames_N9,
                        labels_N = labels_N,
                        data_ids_N = data_ids_N,
                        images_NF = images_NF,
                        rec_images_NF = rec_images_NF)
    elif output_filepath[-5:] == '.hdf5':

        dt = np.dtype([
            ('res_id','S50', (6)),
            ('invariant', 'f4', (invariants_ND.shape[1])),
            ('learned_frame', 'f4', (3, 3)),
            ('label', 'i4', (1)),
            ('original_zgram', 'f4', (images_NF.shape[1])),
            ('reconstructed_zgram', 'f4', (rec_images_NF.shape[1]))
        ])

        def unstringify(stringified):
            return np.array(list(map(lambda x: x.encode('utf-8'), stringified.split('_'))))

        with h5py.File(output_filepath, 'w') as f:
            f.create_dataset('hvae_inference',
                         shape=(invariants_ND.shape[0],),
                         dtype=dt)
            
            data_ids_N = np.array(list(map(unstringify, data_ids_N)))
            learned_frames_N33 = learned_frames_N9.reshape(-1, 3, 3)

            for n in range(invariants_ND.shape[0]):
                f['hvae_inference'][n] = (data_ids_N[n], invariants_ND[n], learned_frames_N33[n], labels_N[n], images_NF[n], rec_images_NF[n],)
            
            if data_filepath[-5:] == '.hdf5':
                with h5py.File(data_filepath, 'r') as f_in:
                    for dataset_name in f_in.keys():
                        if dataset_name != input_dataset_name:
                            f_in.copy(f_in[dataset_name], f)
    
    else:
        raise NotImplementedError()

    if verbose: print('Done saving')
    sys.stdout.flush()


def finetune(model, hparams, n_finetuning_epochs, test_dataloader, device):

    if hparams['weight_decay']:
        optimizer_all = torch.optim.Adam(model.parameters(), lr=hparams['lr'], weight_decay=1e-5)
    else:
        optimizer_all = torch.optim.Adam(model.parameters(), lr=hparams['lr'])

    optimizers = [optimizer_all]
        
    def optimizing_step(x_reconst_loss: Tensor, kl_divergence: Tensor,
                        x_lambda: float, kl_lambda: float,
                        optimizers: List):
        if len(optimizers) == 1: # just one optimizer with all parameters
            optimizer = optimizers[0]
            loss = x_lambda * x_reconst_loss + kl_lambda * kl_divergence
            loss.backward()
            optimizer.step()
        return loss

    if isinstance(hparams['lambdas'], str):
        hparams['lambdas'] = list(map(float, hparams['lambdas'].split(',')))
    
    x_lambda, kl_lambda = hparams['lambdas']

    for epoch in range(n_finetuning_epochs):

        temp_loss = []
        for i, (X, X_vec, y, (rot, data_ids)) in enumerate(test_dataloader):
            X = put_dict_on_device(X, device)
            X_vec = X_vec.float().to(device)

            frame = rot.float().view(-1, 3, 3).to(device)

            for optimizer in optimizers:
                optimizer.zero_grad()
            model.train()
            x_reconst_loss, kl_divergence, _, (mean, log_var) = model(X, x_vec=X_vec, frame=frame)

            total_loss = optimizing_step(x_reconst_loss, kl_divergence,
                                            x_lambda, kl_lambda,
                                            optimizers)
                                            
            total_loss_with_final_kl = x_lambda * x_reconst_loss + hparams['lambdas'][1] * kl_divergence
            temp_loss.append(total_loss_with_final_kl.item())

            if i % 25 == 0:
                print('Epoch: %d, Step: %d, Loss: %f' % (epoch, i, np.mean(temp_loss)))
                sys.stdout.flush()
                temp_loss = []
        
        # reduce learning rate by one order of magnitude after each epoch
        print('Reducing learning rate by one order of magnitude ...')
        sys.stdout.flush()
        for optimizer in optimizers:
            for g in optimizer.param_groups:
                g['lr'] *= 0.1
    
    return model

    

def hvae_standard_evaluation(experiment_dir: str,
                             model_name: str = 'lowest_total_loss_with_final_kl_model'):

    device = 'cpu'

    with open(os.path.join(experiment_dir, 'hparams.json'), 'r') as f:
        hparams = json.load(f)

    # filter by desired lmax and channels
    OnRadialFunctions = ZernickeRadialFunctions(hparams['rcut'], hparams['rmax']+1, hparams['lmax'], complex_sph = False)
    rst = RadialSphericalTensor(hparams['rmax']+1, OnRadialFunctions, hparams['lmax'], 1, 1)
    if 'n_channels' in hparams:
        mul_rst = MultiChannelRadialSphericalTensor(rst, hparams['n_channels'])
    else:
        mul_rst = MultiChannelRadialSphericalTensor(rst, len(hparams['channels']))
    data_irreps = o3.Irreps(str(mul_rst))

    ls_indices = torch.cat([torch.tensor([l]).repeat(2*l+1) for l in data_irreps.ls])

    full_cosine_loss_fn = eval(NAME_TO_LOSS_FN['cosine'])(data_irreps, device)

    per_l_cosine_loss_fn_dict = {}
    for irr in data_irreps:
        per_l_cosine_loss_fn_dict[str(irr.ir)] = eval(NAME_TO_LOSS_FN['cosine'])(o3.Irreps(str(irr)), device)
    
    arrays = np.load(os.path.join(experiment_dir, 'test_data_results-%s.npz' % (model_name)))
    invariants_ND = arrays['invariants_ND']
    learned_frames_N9 = arrays['learned_frames_N9']
    labels_N = arrays['labels_N']
    # rotations_N9 = arrays['rotations_N9']
    images_NF = torch.tensor(arrays['images_NF'])
    rec_images_NF = torch.tensor(arrays['rec_images_NF'])
    try:
        data_ids_N = arrays['data_ids_N']
    except:
        pass

    N = labels_N.shape[0]

    ## compute powers
    ls_indices = torch.cat([torch.tensor(data_irreps.ls)[torch.tensor(data_irreps.ls) == l].repeat(2*l+1) for l in sorted(list(set(data_irreps.ls)))]).type(torch.float)
    orig_powers_N = torch.einsum('bf,bf,f->b', images_NF, images_NF, 1.0 / (2*ls_indices + 1)).numpy()
    rec_powers_N = torch.einsum('bf,bf,f->b', rec_images_NF, rec_images_NF, 1.0 / (2*ls_indices + 1)).numpy()

    full_cosines_N = []
    for n in tqdm(range(N)):
        full_cosines_N.append(full_cosine_loss_fn(rec_images_NF[n, :], images_NF[n, :]).item())
    full_cosines_N = np.array(full_cosines_N)

    # compute per-l-value cosine loss
    per_l_cosines_N_dict = {}
    for irr in data_irreps:
        temp_l_cosines_N = []
        for n in tqdm(range(N)):
            temp_l_cosines_N.append(per_l_cosine_loss_fn_dict[str(irr.ir)](rec_images_NF[n, ls_indices == irr.ir.l], images_NF[n, ls_indices == irr.ir.l]).item())
        per_l_cosines_N_dict[str(irr.ir)] = np.array(temp_l_cosines_N)
    
    # make directories if they do not exists
    if not os.path.exists(os.path.join(experiment_dir, 'loss_distributions')):
        os.mkdir(os.path.join(experiment_dir, 'loss_distributions'))
    
    if not os.path.exists(os.path.join(experiment_dir, 'latent_space_viz')):
        os.mkdir(os.path.join(experiment_dir, 'latent_space_viz'))
    
    if not os.path.exists(os.path.join(experiment_dir, 'latent_space_classification')):
        os.mkdir(os.path.join(experiment_dir, 'latent_space_classification'))
    

    ## plot the powers in a scatterplot 
    from scipy.stats import pearsonr, spearmanr
    coeff = np.polyfit(orig_powers_N, rec_powers_N, deg=1)[0]
    y_fit = coeff * orig_powers_N

    sp_r = spearmanr(orig_powers_N, rec_powers_N)
    pe_r = pearsonr(orig_powers_N, rec_powers_N)

    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', ls='', color='darkblue', markerfacecolor='darkblue', label='SP-R: %.3f, p-val: %.3f' % (sp_r[0], sp_r[1])),
                    Line2D([0], [0], marker='o', ls='', color='darkblue', markerfacecolor='darkblue', label='PE-R: %.3f, p-val: %.3f' % (pe_r[0], pe_r[1]))]

    plt.scatter(orig_powers_N, rec_powers_N, color='darkblue')
    plt.plot(orig_powers_N, y_fit, color='darkblue')
    plt.title('Total Power')
    plt.xlabel('Original Signal')
    plt.ylabel('Reconstructed Signal')
    plt.legend(handles=legend_elements)
    plt.savefig(os.path.join(experiment_dir, 'latent_space_viz/total_power_comparison-%s.png' % (model_name)))
    plt.close()

    # plot the distribution of cosine loss (histogram)
    plt.figure(figsize=(10, 6))
    plt.hist(full_cosines_N, label='Mean = %.3f' % (np.mean(full_cosines_N)))
    plt.xlabel('Cosine loss')
    plt.ylabel('Count')
    plt.title('Full signal')
    plt.xlim([0, 2])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_dir, 'loss_distributions/full_signal_cosines-%s.png' % (model_name)))
    plt.close()

    # plot the distribution of cosine loss per label (10 histograms)
    nrows = 4
    
    fig, axs = plt.subplots(figsize=(20, nrows*4), nrows=nrows, ncols=5, sharex=True, sharey=True)
    axs = axs.flatten()

    for l_i, label in enumerate(sorted(list(set(list(labels_N))))):
        axs[l_i].hist(full_cosines_N[labels_N == label], label='Mean = %.3f' % (np.mean(full_cosines_N[labels_N == label])))
        axs[l_i].set_xlabel('Cosine loss')
        axs[l_i].set_ylabel('Count')
        axs[l_i].set_title('Label = %s' % (protein_naming.ind_to_aa[label]))
        axs[l_i].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_dir, 'loss_distributions/full_signal_cosines_per_label-%s.png' % (model_name)))


    # plot the distribution of cosine loss per l-value (lmax+1 histograms)
    fig, axs = plt.subplots(figsize=(16, 12), nrows=3, ncols=4, sharex=True, sharey=True)
    axs = axs.flatten()

    for l_i, irr in enumerate(data_irreps):
        axs[l_i].hist(per_l_cosines_N_dict[str(irr.ir)], label='Mean = %.3f' % (np.mean(per_l_cosines_N_dict[str(irr.ir)])))
        axs[l_i].set_xlabel('Cosine loss')
        axs[l_i].set_ylabel('Count')
        axs[l_i].set_title('l = %d' % (irr.ir.l))
        axs[l_i].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_dir, 'loss_distributions/per_l_cosines-%s.png' % (model_name)))


    # umap embedding of the latent space
    lower_dim_invariants_N2 = umap.UMAP(random_state=42).fit_transform(invariants_ND)

    MARKER_SCALING = 0.4
    ALPHA = 0.4
    
    GOLDEN_RATIO = (1 + 5.0**0.5) / 2

    HEIGHT = 5.0
    WIDTH = HEIGHT * GOLDEN_RATIO

    colors_20 = plt.get_cmap('tab20').colors
    colors_aa_N = list(map(lambda i: colors_20[i], labels_N))
    from experiments.protein_neighborhoods.src.utils.protein_naming import ind_to_aa
    aa_legend_elements = [Line2D([0], [0], marker='o', ls='', color=colors_20[label], markerfacecolor=colors_20[label], label='%s' % (ind_to_aa[label])) for label in list(sorted(list(set(list(labels_N)))))]

    # aa type umap
    plt.figure(figsize=(WIDTH, HEIGHT))
    plt.scatter(lower_dim_invariants_N2[:, 0], lower_dim_invariants_N2[:, 1], c=colors_aa_N, alpha=ALPHA, edgecolors='none', s=(mpl.rcParams['lines.markersize']*MARKER_SCALING)**2, )
    plt.axis('off')
    plt.legend(handles=aa_legend_elements)
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_dir, 'latent_space_viz/aa_type_umap-%s-split=test.png' % (model_name)))
    plt.close()


    # pretty secondary structure umap
    COLORS_10 = plt.get_cmap('tab10').colors
    COLORS_SEC_STRUCT = [
        COLORS_10[0],
        COLORS_10[7],
        COLORS_10[6]
    ]
    sec_struct_N = [data_id.split('_')[5] for data_id in data_ids_N]
    sec_struct_sorted_set = list(sorted(list(set(sec_struct_N))))
    sec_struct_idx_dict = {}
    for i, sec_struct in enumerate(sec_struct_sorted_set):
        sec_struct_idx_dict[sec_struct] = i
    sec_struct_idxs_N = []
    for sec_struct in sec_struct_N:
        sec_struct_idxs_N.append(sec_struct_idx_dict[sec_struct])
    sec_struct_colors_N = list(map(lambda i: COLORS_SEC_STRUCT[i], sec_struct_idxs_N))
    sec_struct_name_dict = {
        'H': '$\\alpha$-helix',
        'E': '$\\beta$-sheet',
        'L': 'loop'
    }
    sec_struct_legend_elements = [Line2D([0], [0], marker='o', markersize=12.0, ls='', color=COLORS_SEC_STRUCT[sec_struct_idx_dict[label]], markerfacecolor=COLORS_SEC_STRUCT[sec_struct_idx_dict[label]], label='%s' % (sec_struct_name_dict[label])) for label in list(sorted(list(set(list(sec_struct_N)))))]

    # individual plots with categorical labels
    plt.figure(figsize=(WIDTH, HEIGHT))
    plt.scatter(lower_dim_invariants_N2[:, 0], lower_dim_invariants_N2[:, 1], c=np.array(sec_struct_colors_N), alpha=ALPHA, edgecolors='none', s=(mpl.rcParams['lines.markersize']*MARKER_SCALING)**2)
    # plt.legend(handles=legend_handles, prop={'size': 14.5})
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_dir, 'latent_space_viz/__pretty_umap__secondary_structure_%s-split=test.png' % (model_name)), bbox_inches='tight')
    # plt.savefig(os.path.join(experiment_dir, 'latent_space_viz/__pretty_umap__secondary_structure_%s-split=test.pdf' % (model_name)), bbox_inches='tight')
    plt.close()

    # save legend separately
    figLegend = pylab.figure(figsize = (6.0, 0.6))
    figLegend.legend(handles=sec_struct_legend_elements, loc='center', ncol=len(sec_struct_legend_elements), prop={'size': 18})
    figLegend.savefig(os.path.join(experiment_dir, 'latent_space_viz/__legend__secondary_structure.png'))
    # figLegend.savefig(os.path.join(experiment_dir, 'latent_space_viz/__legend__secondary_structure.pdf'))