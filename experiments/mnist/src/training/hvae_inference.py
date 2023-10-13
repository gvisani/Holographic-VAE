
import os, sys
import gzip, pickle
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pylab
import pandas as pd

import torch
import torch.nn.functional as F
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
from holographic_vae.utils.loss_functions import *

from experiments.mnist.src.utils.data import SphericalMNISTDataset

from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import accuracy_score, classification_report, homogeneity_score, completeness_score, silhouette_score

def purity_score(labels_true, labels_pred):
    cont_matrix = contingency_matrix(labels_true, labels_pred)
    return np.sum(np.amax(cont_matrix, axis=0)) / np.sum(cont_matrix)

def latent_space_prediction(train_invariants_ND, train_labels_N, valid_invariants_ND, valid_labels_N, classifier='LR', optimize_hyps=False, data_percentage=100):
    
    if classifier == 'LR':
        estimator = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
        hyperparams = {'C': [0.1, 0.5, 1.0, 5.0, 10.0]}
    elif classifier == 'RF':
        estimator = RandomForestClassifier()
        hyperparams = {'min_samples_leaf': [1, 2, 5, 10, 20, 50, 100]}
    
    if optimize_hyps:
        model = GridSearchCV(estimator, hyperparams)
    else:
        model = estimator
    
    model = model.fit(train_invariants_ND, train_labels_N)
    
    predictions = model.predict_proba(valid_invariants_ND)
    onehot_predictions = np.argmax(predictions, axis=1)
    
    return classification_report(valid_labels_N, onehot_predictions, output_dict=True)


def hvae_inference(experiment_dir: str,
                    split: str = 'test',
                    model_name: str = 'lowest_total_loss_with_final_kl_model',
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

    from experiments.mnist.src.utils.data import load_data
    output_filepath = os.path.join(experiment_dir, 'evaluation_results-split={}-model_name={}.npz'.format(split, model_name))
    datasets, data_irreps = load_data(hparams, splits=[split])
    test_dataset = datasets[split]
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, generator=rng, shuffle=False, drop_last=False)
    
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
    model = H_VAE(data_irreps, w3j_matrices, hparams['model_hparams'], device, normalize_input_at_runtime=False).to(device)
    model.load_state_dict(torch.load(os.path.join(experiment_dir, model_name + '.pt'), map_location=torch.device(device)))

    num_params = 0
    for param in model.parameters():
        num_params += torch.flatten(param.data).shape[0]
    if verbose: print('There are %d parameters' % (num_params))
    sys.stdout.flush()

    model.eval()
    invariants, labels, learned_frames, images, rec_images = [], [], [], [], []
    
    if loading_bar:
        loadind_bar = tqdm
    else:
        loadind_bar = lambda x: x

    for i, (X, X_vec, y, rot) in loadind_bar(enumerate(test_dataloader)):
        X = put_dict_on_device(X, device)
        rot = rot.view(-1, 3, 3).float().to(device)

        (z_mean, _), learned_frame = model.encode(X)

        z = z_mean

        if model.is_conditional:
            conditioned_z = model.condition_latent_space(z, F.one_hot(y.long(), num_classes=10).float().to(device))
        else:
            conditioned_z = z

        if hparams['model_hparams']['learn_frame']:
            x_reconst = model.decode(conditioned_z, learned_frame)
        else:
            x_reconst = model.decode(conditioned_z, rot)

        invariants.append(z.detach().cpu().numpy())

        if hparams['model_hparams']['learn_frame']:
            learned_frames.append(learned_frame.reshape(-1, 1, 9).squeeze(1).detach().cpu().numpy())
        else:
            learned_frames.append(rot.reshape(-1, 1, 9).squeeze(1).cpu().numpy())
        
        labels.append(y.cpu().numpy())
        images.append(X_vec.detach().cpu().numpy())
        rec_images.append(make_vec(x_reconst).detach().cpu().numpy())
        
    invariants_ND = np.vstack(invariants)
    learned_frames_N9 = np.vstack(learned_frames)
    labels_N = np.hstack(labels)
    images_NF = np.vstack(images)
    rec_images_NF = np.vstack(rec_images)
    
    cosine_loss_fn = eval(NAME_TO_LOSS_FN['cosine'])(data_irreps, device)
    cosine_loss = cosine_loss_fn(torch.tensor(images_NF).float().to(device), torch.tensor(rec_images_NF).float().to(device)).item()
    mse_loss = torch.nn.functional.mse_loss(torch.tensor(images_NF).float().to(device), torch.tensor(rec_images_NF).float().to(device)).item()
    if verbose: print('Cosine loss: {:.3f}'.format(cosine_loss))
    if verbose: print('MSE loss: {:.8f}'.format(mse_loss))

    if verbose: print('Done running model, saving to %s ...' % (output_filepath))
    sys.stdout.flush()

    np.savez(output_filepath,
                    invariants_ND = invariants_ND,
                    learned_frames_N9 = learned_frames_N9,
                    labels_N = labels_N,
                    images_NF = images_NF,
                    rec_images_NF = rec_images_NF)

    if verbose: print('Done saving')
    sys.stdout.flush()

    

def hvae_standard_evaluation(experiment_dir: str,
                             split: str = 'test',
                             model_name: str = 'lowest_total_loss_with_final_kl_model'):

    device = 'cpu'

    with open(os.path.join(experiment_dir, 'hparams.json'), 'r') as f:
        hparams = json.load(f)

    data_irreps = o3.Irreps.spherical_harmonics(hparams['lmax'], 1)
    ls_indices = torch.cat([torch.tensor([l]).repeat(2*l+1) for l in data_irreps.ls])

    full_cosine_loss_fn = eval(NAME_TO_LOSS_FN['cosine'])(data_irreps, device)

    per_l_cosine_loss_fn_dict = {}
    for irr in data_irreps:
        per_l_cosine_loss_fn_dict[str(irr.ir)] = eval(NAME_TO_LOSS_FN['cosine'])(o3.Irreps(str(irr)), device)
    
    arrays = np.load(os.path.join(experiment_dir, 'evaluation_results-split={}-model_name={}.npz'.format(split, model_name)))
    invariants_ND = arrays['invariants_ND']
    learned_frames_N9 = arrays['learned_frames_N9']
    labels_N = arrays['labels_N']
    images_NF = torch.tensor(arrays['images_NF'])
    rec_images_NF = torch.tensor(arrays['rec_images_NF'])

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
        axs[l_i].set_title('Label = %d' % (label))
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


    # umap embedding of the latent space, colored by label
    lower_dim_invariants_N2 = umap.UMAP(random_state=42).fit_transform(invariants_ND)

    MARKER_SCALING = 0.7
    ALPHA = 0.5
    
    GOLDEN_RATIO = (1 + 5.0**0.5) / 2

    HEIGHT = 5.0
    WIDTH = HEIGHT * GOLDEN_RATIO

    colors_10 = plt.get_cmap('tab10').colors
    colors_label_N = list(map(lambda i: colors_10[i], labels_N))
    label_legend_elements = [Line2D([0], [0], marker='o', ls='', color=colors_10[label], markerfacecolor=colors_10[label], label='%d' % (label)) for label in list(sorted(list(set(list(labels_N)))))]

    plt.figure(figsize=(WIDTH, HEIGHT))
    plt.scatter(lower_dim_invariants_N2[:, 0], lower_dim_invariants_N2[:, 1], c=colors_label_N, alpha=ALPHA, edgecolors='none', s=(mpl.rcParams['lines.markersize']*MARKER_SCALING)**2, )
    plt.axis('off')
    plt.legend(handles=label_legend_elements)
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_dir, 'latent_space_viz/umap-split=%s-%s.png' % (split, model_name)))
    plt.close()


def classification_and_clustering_in_latent_space(experiment_dir: str,
                                                    model_name: str = 'lowest_total_loss_with_final_kl_model',
                                                    verbose: bool = False,
                                                    loading_bar: bool = True,
                                                    batch_size: int = 100):
    
    ## make data
    if not os.path.exists(os.path.join(experiment_dir, 'evaluation_results-split={}-model_name={}.npz'.format('train', model_name))):
        hvae_inference(experiment_dir, split='train', model_name=model_name, verbose=verbose, loading_bar=loading_bar, batch_size=batch_size)
    if not os.path.exists(os.path.join(experiment_dir, 'evaluation_results-split={}-model_name={}.npz'.format('valid', model_name))):
        hvae_inference(experiment_dir, split='valid', model_name=model_name, verbose=verbose, loading_bar=loading_bar, batch_size=batch_size)
    if not os.path.exists(os.path.join(experiment_dir, 'evaluation_results-split={}-model_name={}.npz'.format('test', model_name))):
        hvae_inference(experiment_dir, split='test', model_name=model_name, verbose=verbose, loading_bar=loading_bar, batch_size=batch_size)

    ## load data
    train_arrays = np.load(os.path.join(experiment_dir, 'evaluation_results-split={}-model_name={}.npz'.format('train', model_name)))
    valid_arrays = np.load(os.path.join(experiment_dir, 'evaluation_results-split={}-model_name={}.npz'.format('valid', model_name)))
    train_invariants_ND = np.vstack([train_arrays['invariants_ND'], valid_arrays['invariants_ND']])
    train_labels_N = np.hstack([train_arrays['labels_N'], valid_arrays['labels_N']])

    test_arrays = np.load(os.path.join(experiment_dir, 'evaluation_results-split={}-model_name={}.npz'.format('test', model_name)))
    test_invariants_ND = test_arrays['invariants_ND']
    test_labels_N = test_arrays['labels_N']

    
    if not os.path.exists(os.path.join(experiment_dir, 'latent_space_classification')):
        os.mkdir(os.path.join(experiment_dir, 'latent_space_classification'))

    ## classification
    pd.DataFrame(latent_space_prediction(train_invariants_ND, train_labels_N, test_invariants_ND, test_labels_N, classifier='LR', optimize_hyps=True, data_percentage=100)).to_csv(os.path.join(experiment_dir, 'latent_space_classification', 'latent_space_classification-classifier=%s-model_name=%s.csv' % ('LR', model_name)))
    pd.DataFrame(latent_space_prediction(train_invariants_ND, train_labels_N, test_invariants_ND, test_labels_N, classifier='RF', optimize_hyps=True, data_percentage=100)).to_csv(os.path.join(experiment_dir, 'latent_space_classification', 'latent_space_classification-classifier=%s-model_name=%s.csv' % ('RF', model_name)))

    ## clustering
    labels_pred_N = KMeans(n_clusters=10, random_state=1234567890, verbose=0).fit_predict(test_invariants_ND)

    homogeneity = homogeneity_score(test_labels_N, labels_pred_N)
    completeness = completeness_score(test_labels_N, labels_pred_N)
    purity = purity_score(test_labels_N, labels_pred_N)
    silhouette = silhouette_score(test_invariants_ND, labels_pred_N)

    table = {
        'Homogeneity': [homogeneity],
        'Completeness': [completeness],
        'Purity': [purity],
        'Silhouette': [silhouette]
    }

    pd.DataFrame(table).to_csv(os.path.join(experiment_dir, 'latent_space_classification', 'quality_of_clustering_metrics_default_classes-model_name=%s.csv' % (model_name)))




def hvae_reconstruction_tests(experiment_dir: str,
                         grid_dir: str = './data/',
                         n_samples: int = 5,
                         split: str = 'test',
                         model_name: str = 'lowest_total_loss_with_final_kl_model',
                         seed: int = 123456789,
                         verbose: bool = True):
    
    from experiments.mnist.src.preprocessing import real_sph_ift
    from holographic_vae.so3.functional import get_wigner_D_from_rot_matrix, rotate_signal, make_vec, put_dict_on_device, make_dict
    
    # get hparams from json
    with open(os.path.join(experiment_dir, 'hparams.json'), 'r') as f:
        hparams = json.load(f)
    
    data_irreps = o3.Irreps.spherical_harmonics(hparams['lmax'], 1)

    # load grid
    with gzip.open(os.path.join(grid_dir, 'ba_grid-b=%d.gz' % (hparams['bw'])), 'rb') as f:
        ba_grid = pickle.load(f)

        # flatten grid
        ba_grid = list(ba_grid)
        for i in range(2):
            ba_grid[i] = ba_grid[i].flatten()
        ba_grid = tuple(ba_grid)

    # setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if verbose: print('Running on %s.' % (device))

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
    model = H_VAE(data_irreps, w3j_matrices, hparams['model_hparams'], device, normalize_input_at_runtime=False).to(device)
    model.load_state_dict(torch.load(os.path.join(experiment_dir, model_name + '.pt'), map_location=torch.device(device)))
    model.eval()

    num_params = 0
    for param in model.parameters():
        num_params += torch.flatten(param.data).shape[0]
    if verbose: print('There are %d parameters' % (num_params))
    sys.stdout.flush()


    # load results (output of running `hvae_inference()`)
    arrays = np.load(os.path.join(experiment_dir, 'evaluation_results-split={}-model_name={}.npz'.format(split, model_name)))
    invariants_ND = arrays['invariants_ND']
    learned_frames_N9 = arrays['learned_frames_N9']
    labels_N = arrays['labels_N']
    images_NF = torch.tensor(arrays['images_NF'])
    rec_images_NF = torch.tensor(arrays['rec_images_NF'])


    N = labels_N.shape[0]

    if not os.path.exists(os.path.join(experiment_dir, 'reconstructions')):
        os.mkdir(os.path.join(experiment_dir, 'reconstructions'))

    IMAGE_SIZE = 4

    for label in sorted(list(set(list(labels_N)))):
        idxs_label = np.arange(labels_N.shape[0])[labels_N == label]
        idxs_to_show = np.random.default_rng(seed).choice(idxs_label, size=n_samples)

        imgs = images_NF[idxs_to_show]
        imgs_rec = rec_images_NF[idxs_to_show]

        frames = learned_frames_N9[idxs_to_show]
        temp_imgs = []
        temp_imgs_rec = []
        for i in range(n_samples):
            undo_rot_wigner = get_wigner_D_from_rot_matrix(max(data_irreps.ls), torch.tensor(frames[i]).view(3, 3))
            temp_imgs.append(real_sph_ift(rotate_signal(imgs[i].unsqueeze(0), data_irreps, undo_rot_wigner), ba_grid, max(data_irreps.ls))[0])
            temp_imgs_rec.append(real_sph_ift(rotate_signal(imgs_rec[i].unsqueeze(0), data_irreps, undo_rot_wigner), ba_grid, max(data_irreps.ls))[0])
        imgs = temp_imgs
        imgs_rec = temp_imgs_rec

        fig, axs = plt.subplots(figsize=(n_samples*IMAGE_SIZE, 2*IMAGE_SIZE), ncols=n_samples, nrows=2, sharex=True, sharey=True)

        for i, (img, img_rec) in enumerate(zip(imgs, imgs_rec)):
            axs[0][i].imshow(img.reshape(60, 60))
            axs[1][i].imshow(img_rec.reshape(60, 60))
        
        plt.tight_layout()
        plt.savefig(os.path.join(experiment_dir, 'reconstructions/example_reconstructions-%s-label=%d-split=%s-seed=%d.png' % (model_name, label, split, seed)))
        plt.close()


    for label in sorted(list(set(list(labels_N)))):
        idxs_label = np.arange(labels_N.shape[0])[labels_N == label]
        idxs_to_show = np.random.default_rng(seed).choice(idxs_label, size=n_samples)

        imgs = images_NF[idxs_to_show]
        imgs_rec = rec_images_NF[idxs_to_show]

        frames = learned_frames_N9[idxs_to_show]
        temp_imgs = []
        temp_imgs_rec = []
        for i in range(n_samples):
            undo_rot_wigner = get_wigner_D_from_rot_matrix(max(data_irreps.ls), torch.tensor(frames[i]).view(3, 3))
            temp_imgs.append(real_sph_ift(rotate_signal(imgs[i].unsqueeze(0), data_irreps, undo_rot_wigner), ba_grid, max(data_irreps.ls))[0])
            temp_imgs_rec.append(real_sph_ift(rotate_signal(imgs_rec[i].unsqueeze(0), data_irreps, undo_rot_wigner), ba_grid, max(data_irreps.ls))[0])
        imgs = temp_imgs
        imgs_rec = temp_imgs_rec

        for i, (img, img_rec) in enumerate(zip(imgs, imgs_rec)):
            plt.figure(figsize=(IMAGE_SIZE, IMAGE_SIZE))
            plt.imshow(img.reshape(60, 60))
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(experiment_dir, 'reconstructions/__true_image-n=%d-label=%d-split=%s-seed=%d.png' % (i, label, split, seed)))
            plt.savefig(os.path.join(experiment_dir, 'reconstructions/__true_image-n=%d-label=%d-split=%s-seed=%d.pdf' % (i, label, split, seed)))
            plt.close()

            plt.figure(figsize=(IMAGE_SIZE, IMAGE_SIZE))
            plt.imshow(img_rec.reshape(60, 60))
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(experiment_dir, 'reconstructions/__rec_image-n=%d-label=%d-split=%s-seed=%d.png' % (i, label, split, seed)))
            plt.savefig(os.path.join(experiment_dir, 'reconstructions/__rec_image-n=%d-label=%d-split=%s-seed=%d.pdf' % (i, label, split, seed)))
            plt.close()
    



    ## example reconstruction of test data, with visual equivariance tests
    print('Plotting some reconstructions...')

    # (0) get some test signal
    torch.manual_seed(seed)
    idxs = torch.randint(N, size=(n_samples,))
    signal_orig = images_NF[idxs].float()
    y_orig = torch.tensor(labels_N)[idxs]

    # get some random rotation
    rot_matrix = e3nn.o3.rand_matrix(1).float().view(-1, 1, 9).squeeze().unsqueeze(0)
    wigner = get_wigner_D_from_rot_matrix(max(data_irreps.ls), rot_matrix[0].view(3, 3))

    # (1) rotate original signal
    signal_rot = rotate_signal(signal_orig, data_irreps, wigner)

    (z_mean_orig, _), learned_frame_orig = model.encode(put_dict_on_device(make_dict(signal_orig, data_irreps), device))
    (z_mean_rot, _), learned_frame_rot = model.encode(put_dict_on_device(make_dict(signal_rot, data_irreps), device))

    if model.is_conditional:
        z_mean_orig = model.condition_latent_space(z_mean_orig, F.one_hot(y_orig.long()).float().to(device))
        z_mean_rot = model.condition_latent_space(z_mean_rot, F.one_hot(y_orig.long()).float().to(device))

    # rotate learned frame of original input
    learned_frame_orig_rot = rotate_signal(learned_frame_orig.reshape(-1, 1, 9).squeeze(1).detach().cpu(), o3.Irreps('3x1e'), wigner)

    learned_frame_orig = learned_frame_orig.reshape(-1, 3, 3)
    learned_frame_rot = learned_frame_rot.reshape(-1, 3, 3)
    learned_frame_orig_rot = learned_frame_orig_rot.reshape(-1, 3, 3)


    # (2) original reconstruction
    x_reconst_orig = make_vec(model.decode(z_mean_orig.to(device), learned_frame_orig.to(device)))

    # (3) reconstruction with rotated input
    x_reconst_rot = make_vec(model.decode(z_mean_rot.to(device), learned_frame_rot.to(device)))
    
    # (4) original reconstruction, rotated afterwards
    x_reconst_orig_rot = rotate_signal(x_reconst_orig.detach().cpu(), data_irreps, wigner)

    # (5) reconstruction with rotated input, but using latent vector of original signal (check that latents are invariant)
    x_reconst_rot_with_orig_z = make_vec(model.decode(z_mean_orig.to(device), learned_frame_rot.to(device)))

    # (6) reconstruction of original input, with rotated frame
    x_reconst_orig_rot_frame = make_vec(model.decode(z_mean_orig.to(device), learned_frame_orig_rot.to(device)))

    # (0) and (2) should be as close as possible
    # (3), (4), (5), (6) should be identical to one another up to numerical error, and as close as possible to (1)


    signal_orig_inv = real_sph_ift(signal_orig.cpu().detach(), ba_grid, max(data_irreps.ls))
    signal_rot_inv = real_sph_ift(signal_rot.cpu().detach(), ba_grid, max(data_irreps.ls))

    rec_orig_inv = real_sph_ift(x_reconst_orig.cpu().detach(), ba_grid, max(data_irreps.ls))
    rec_rot_inv = real_sph_ift(x_reconst_rot.cpu().detach(), ba_grid, max(data_irreps.ls))
    rec_orig_rot_inv = real_sph_ift(x_reconst_orig_rot.cpu().detach(), ba_grid, max(data_irreps.ls))
    rec_rot_with_orig_z_inv = real_sph_ift(x_reconst_rot_with_orig_z.cpu().detach(), ba_grid, max(data_irreps.ls))
    rec_orig_rot_frame_inv = real_sph_ift(x_reconst_orig_rot_frame.cpu().detach(), ba_grid, max(data_irreps.ls))

    # reconstruction of rotated signal, for visual equivariance test
    fig, axs = plt.subplots(figsize=(20, 4*n_samples), ncols=5, nrows=n_samples, sharex=True, sharey=True)
    axs[0][0].set_title('Rotated\n(fwd/inv)')
    axs[0][1].set_title('Rotating input')
    axs[0][2].set_title('Rotating output')
    axs[0][3].set_title('Rotating input,\nz from unrotated')
    axs[0][4].set_title('Rotating frame')

    for i in range(n_samples):
        axs[i][0].imshow(signal_rot_inv[i].reshape(60, 60))
        axs[i][1].imshow(rec_rot_inv[i].reshape(60, 60))
        axs[i][2].imshow(rec_orig_rot_inv[i].reshape(60, 60))
        axs[i][3].imshow(rec_rot_with_orig_z_inv[i].reshape(60, 60))
        axs[i][4].imshow(rec_orig_rot_frame_inv[i].reshape(60, 60))

    plt.savefig(os.path.join(experiment_dir, 'reconstructions/test_reconstructions_with_rotations-%s-split=%s-seed=%d.png' % (model_name, split, seed)))
    plt.close()


    ## plot some samples in the canonical frame
    ## sample some vectors in the latent space
    ## sample around the prior (zero mean, unit variance)
    if hparams['model_hparams']['is_vae']:

        if not model.is_conditional:
            print('Plotting some samples...')

            ## initialize generator
            generator = torch.Generator().manual_seed(seed)
            
            z = torch.normal(torch.zeros((n_samples*n_samples, hparams['model_hparams']['latent_dim'])), torch.ones((n_samples*n_samples, hparams['model_hparams']['latent_dim'])), generator=generator)
            frame = torch.eye(3).repeat(n_samples*n_samples, 1).float().view(-1, 3, 3).squeeze().to(device)
            x_reconst = model.decode(z.to(device), frame.to(device))
            rec_inv_NF = real_sph_ift(make_vec(x_reconst).cpu().detach(), ba_grid, max(data_irreps.ls))


            for n in range(n_samples**2):
                fig, ax = plt.subplots(figsize=(4, 4), nrows=1, ncols=1)
                ax.imshow(rec_inv_NF[n].reshape(60, 60))
                ax.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(experiment_dir, 'reconstructions/sample_n%d-seed=%d.png' % (n, seed)))
                plt.savefig(os.path.join(experiment_dir, 'reconstructions/sample_n%d-seed=%d.pdf' % (n, seed)))
                plt.close()
            
            fig, axs = plt.subplots(figsize=(4*n_samples, 4*n_samples), nrows=n_samples, ncols=n_samples, sharex=True, sharey=True)
            for n, ax in enumerate(axs.flatten()):
                ax.imshow(rec_inv_NF[n].reshape(60, 60))
                ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(experiment_dir, 'reconstructions/samples-n_samples=%d-seed=%d.png' % (n_samples, seed)))
            plt.savefig(os.path.join(experiment_dir, 'reconstructions/samples-n_samples=%d-seed=%d.pdf' % (n_samples, seed)))
            plt.close()
        
        else:
            print('Plotting some conditional samples...')

            ## initialize generator
            generator = torch.Generator().manual_seed(seed)
            
            z = torch.normal(torch.zeros((10*n_samples, hparams['model_hparams']['latent_dim'])), torch.ones((10*n_samples, hparams['model_hparams']['latent_dim'])), generator=generator)
            frame = torch.eye(3).repeat(10*n_samples, 1).float().view(-1, 3, 3).squeeze().to(device)
            y = torch.arange(10).repeat(n_samples).to(device)
            z = model.condition_latent_space(z.to(device), F.one_hot(y.long(), num_classes=10).float().to(device))
            x_reconst = model.decode(z.to(device), frame.to(device))
            rec_inv_NF = real_sph_ift(make_vec(x_reconst).cpu().detach(), ba_grid, max(data_irreps.ls))
            
            fig, axs = plt.subplots(figsize=(4*n_samples, 4*10), nrows=10, ncols=n_samples, sharex=True, sharey=True)

            for n in range(10*n_samples):
                ax = axs[n % 10, n // 10]
                ax.imshow(rec_inv_NF[n].reshape(60, 60))
                ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(experiment_dir, 'reconstructions/conditional_samples-n_samples=%d-seed=%d.png' % (n_samples, seed)))
            plt.savefig(os.path.join(experiment_dir, 'reconstructions/conditional_samples-n_samples=%d-seed=%d.pdf' % (n_samples, seed)))
            plt.close()



