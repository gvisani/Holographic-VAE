import os, sys
import gzip, pickle
import json
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
import e3nn
from e3nn import o3
from sklearn.metrics import accuracy_score
from scipy.special import softmax
from copy import deepcopy

from typing import *

from holographic_vae.models import H_VAE
from holographic_vae.so3.functional import put_dict_on_device
from holographic_vae.cg_coefficients import get_w3j_coefficients


def hvae_training(experiment_dir: str):
    '''
    Assumes that directory 'experiment_dir' exists and contains json file with data and model hyperprameters 
    '''

    # get hparams from json`
    with open(os.path.join(experiment_dir, 'hparams.json'), 'r') as f:
        hparams = json.load(f)

    # seed the random number generator
    rng = torch.Generator().manual_seed(hparams['seed'])

    # setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Running on %s.' % (device))
    sys.stdout.flush()

    print('Loading data...')
    sys.stdout.flush()
    
    ########## THE CODE BLOCK BELOW MAY BE CHANGED TO ACCOMODATE A DIFFERENT DATA-LOADING PIPELINE ##########

    # get data and make dataloaders
    from experiments.mnist.src.utils.data import load_data
    datasets, data_irreps = load_data(hparams, splits=['train', 'valid'])
    train_dataset, valid_dataset = datasets['train'], datasets['valid']
    train_dataloader = DataLoader(train_dataset, batch_size=hparams['batch_size'], generator=rng, shuffle=True, drop_last=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=hparams['batch_size'], generator=rng, shuffle=True, drop_last=False)

    ########## THIS CODE BLOCK ABOVE MAY BE CHANGED TO ACCOMODATE A DIFFERENT DATA-LOADING PIPELINE ##########
    
    print('Data Irreps: %s' % (str(data_irreps)))
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
    
    # create model
    model = H_VAE(data_irreps, w3j_matrices, hparams['model_hparams'], device, normalize_input_at_runtime=False).to(device)
    
    num_params = 0
    for param in model.parameters():
        num_params += torch.flatten(param.data).shape[0]
    print('There are %d parameters' % (num_params))
    sys.stdout.flush()

    # setup learning algorithm
    
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

    x_lambda, kl_lambda = hparams['lambdas']

    if hparams['lr_scheduler'] is None or hparams['lr_scheduler'] == 'constant':
        lr_scheduler = None
        lr_list = None
    
    elif hparams['lr_scheduler'] == 'reduce_lr_on_plateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_all, mode='min', factor=0.2, patience=5, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)
        lr_list = None

    elif hparams['lr_scheduler'] == 'log_decrease_until_end_of_warmup':
        init_lr_scale = float(('%e' % (hparams['lr'])).split('e')[0])
        init_lr_exponent = int(('%e' % (hparams['lr'])).split('e')[1])
        lr_list = list(init_lr_scale*np.logspace(init_lr_exponent, init_lr_exponent-1, hparams['no_kl_epochs'])) + list(init_lr_scale*np.full(hparams['n_epochs'] - hparams['no_kl_epochs'], float('1e%d' % (init_lr_exponent-1))))
        lr_scheduler = None
    elif hparams['lr_scheduler'] == 'linear_decrease_until_end_of_warmup':
        lr_list = list(np.linspace(hparams['lr'], hparams['lr'] * 0.1, hparams['no_kl_epochs'])) + list(np.full(hparams['n_epochs'] - hparams['no_kl_epochs'], hparams['lr'] * 0.1))
        lr_scheduler = None
    elif hparams['lr_scheduler'] == 'log_decrease_until_end_by_1_OM':
        init_lr_scale = float(('%e' % (hparams['lr'])).split('e')[0])
        init_lr_exponent = int(('%e' % (hparams['lr'])).split('e')[1])
        lr_list = list(init_lr_scale*np.logspace(init_lr_exponent, init_lr_exponent-1, hparams['n_epochs']))
        lr_scheduler = None
    elif hparams['lr_scheduler'] == 'log_decrease_until_end_by_2_OM':
        init_lr_scale = float(('%e' % (hparams['lr'])).split('e')[0])
        init_lr_exponent = int(('%e' % (hparams['lr'])).split('e')[1])
        lr_list = list(init_lr_scale*np.logspace(init_lr_exponent, init_lr_exponent-2, hparams['n_epochs']))
        lr_scheduler = None
    elif hparams['lr_scheduler'] == 'log_decrease_until_end_by_3_OM':
        init_lr_scale = float(('%e' % (hparams['lr'])).split('e')[0])
        init_lr_exponent = int(('%e' % (hparams['lr'])).split('e')[1])
        lr_list = list(init_lr_scale*np.logspace(init_lr_exponent, init_lr_exponent-3, hparams['n_epochs']))
        lr_scheduler = None
    else:
        raise NotImplementedError()


    if hparams['lambdas_schedule'] == 'linear_up_anneal_kl':
        kl_lambda_per_epoch = list(np.zeros(hparams['no_kl_epochs'])) + list(np.linspace(0.0, kl_lambda, hparams['warmup_kl_epochs'])) + list(np.full(hparams['n_epochs'] - hparams['warmup_kl_epochs'] - hparams['no_kl_epochs'], kl_lambda))
        print(kl_lambda_per_epoch, file=sys.stderr)
    
    elif hparams['lambdas_schedule'] is None or hparams['lambdas_schedule'] == 'constant':
        kl_lambda_per_epoch = np.full(hparams['n_epochs'], kl_lambda)
    
    elif hparams['lambdas_schedule'] == 'drop_kl_at_half':
        import math
        kl_lambda_per_epoch = list(np.full(math.floor(hparams['n_epochs']), kl_lambda)) + list(np.full(math.floor(hparams['n_epochs']), 0.0))
    
    else:
        raise NotImplementedError()
    
    # TODO: fix this guy... it still gives nans, I think when all values are masked to zero. I thought this would fix it but idk, have to test it's doing what I think it is
    def mask_dict_entries(dict_tensor, p):

        batch_size = dict_tensor[0].shape[0]

        masks = {}
        batch_is_all_zero = torch.ones((batch_size,)).bool()
        for l, tensor in dict_tensor.items():
            B, F, M = tensor.shape
            masks[l] = torch.rand((B, F)) < p
            for f in range(F):
                batch_is_all_zero = torch.logical_and(batch_is_all_zero, masks[l][:, f])
        
        # unmask the first half of the features if all features have been sampled to be masked, as a safety precaution, since, when all values are zero, we get nan loss
        count = 0
        for l in masks:
            masks[l][batch_is_all_zero, :] = False
            count += 1
            if count == len(masks) // 2:
                break

        newdict = {}
        for l, tensor in dict_tensor.items():
            newtensor = deepcopy(tensor)
            newtensor[masks[l]] = 0.0
            newdict[l] = newtensor
        
        return newdict

    global_record_i = 0
    epoch_start = 0
    lowest_rec_loss = np.inf
    lowest_rec_loss_kl = np.inf
    lowest_total_loss_with_final_kl = np.inf
    lowest_total_loss_with_final_kl_kl = np.inf

    print(lr_list, file=sys.stderr)

    times_per_epoch_to_record = 5
    steps_to_record = len(train_dataloader) // times_per_epoch_to_record
    for epoch in range(epoch_start, hparams['n_epochs']):
        print('Epoch %d/%d' % (epoch+1, hparams['n_epochs']))
        sys.stdout.flush()
        train_sf_rec_loss, train_rec_loss, train_kl, train_sf_reg, train_total_loss, train_total_loss_with_final_kl = [], [], [], [], [], []
        train_mean, train_log_var, train_sf, train_sf_rec = {'Mean': [], 'Min': [], 'Max': []}, {'Mean': [], 'Min': [], 'Max': []}, {'Mean': [], 'Min': [], 'Max': []}, {'Mean': [], 'Min': [], 'Max': []}
        record_i = 1
        kl_lambda = kl_lambda_per_epoch[epoch]

        if hparams['lr_scheduler'] == 'decrease_after_warmup' and hparams['lambdas_schedule'] == 'linear_up_anneal_kl' and epoch == hparams['warmup_kl_epochs']: # reduce learning rate after kl warmup
            for optimizer in optimizers:
                for g in optimizer.param_groups:
                    g['lr'] *= 0.1

        elif hparams['lr_scheduler'] == 'log_decrease_until_end_of_warmup':
            for optimizer in optimizers:
                for g in optimizer.param_groups:
                    g['lr'] = lr_list[epoch]

        
        start_time = time.time()
        for i, (X, X_vec, y, rot) in enumerate(train_dataloader):
            X = put_dict_on_device(X, device)
            X_vec = X_vec.float().to(device)
            # y = y.float().to(device)

            if 'noise' in hparams:
                if hparams['noise'] > 0:
                    X = mask_dict_entries(X, hparams['noise'])
                    X = put_dict_on_device(X, device)

            frame = rot.float().view(-1, 3, 3).to(device)

            for optimizer in optimizers:
                optimizer.zero_grad()
            model.train()
            x_reconst_loss, kl_divergence, _, (mean, log_var) = model(X, x_vec=X_vec, frame=frame, c=F.one_hot(y.long(), num_classes=10).float().to(device))

            total_loss = optimizing_step(x_reconst_loss, kl_divergence,
                                            x_lambda, kl_lambda,
                                            optimizers)
            total_loss_with_final_kl = x_lambda * x_reconst_loss + hparams['lambdas'][1] * kl_divergence

            train_total_loss.append(total_loss.item())
            train_total_loss_with_final_kl.append(total_loss_with_final_kl.item())
            train_rec_loss.append(x_reconst_loss.item())
            train_kl.append(kl_divergence.item())
            for key, stat_func in zip(['Mean', 'Min', 'Max'], [np.mean, np.min, np.max]):
                train_mean[key].append(stat_func(mean.cpu().detach().numpy(), axis=-1))
                train_log_var[key].append(stat_func(log_var.cpu().detach().numpy(), axis=-1))


            if i % steps_to_record == (steps_to_record - 1):
                valid_sf_rec_loss, valid_rec_loss, valid_kl, valid_sf_reg, valid_total_loss, valid_total_loss_with_final_kl = [], [], [], [], [], []
                valid_mean, valid_log_var, valid_sf, valid_sf_rec = {'Mean': [], 'Min': [], 'Max': []}, {'Mean': [], 'Min': [], 'Max': []}, {'Mean': [], 'Min': [], 'Max': []}, {'Mean': [], 'Min': [], 'Max': []}
                for j, (X, X_vec, y, rot) in enumerate(valid_dataloader):
                    X = put_dict_on_device(X, device)
                    X_vec = X_vec.float().to(device)
                    # y = y.float().to(device)

                    frame = rot.float().view(-1, 3, 3).to(device)

                    model.eval()
                    x_reconst_loss, kl_divergence, _, (mean, log_var) = model(X, x_vec=X_vec, frame=frame, c=F.one_hot(y.long(), num_classes=10).float().to(device))

                    total_loss = x_lambda * x_reconst_loss + kl_lambda * kl_divergence
                    total_loss_with_final_kl = x_lambda * x_reconst_loss + hparams['lambdas'][1] * kl_divergence


                    valid_total_loss.append(total_loss.item())
                    valid_total_loss_with_final_kl.append(total_loss_with_final_kl.item())
                    valid_rec_loss.append(x_reconst_loss.item())
                    valid_kl.append(kl_divergence.item())
                    for key, stat_func in zip(['Mean', 'Min', 'Max'], [np.mean, np.min, np.max]):
                        valid_mean[key].append(stat_func(mean.cpu().detach().numpy(), axis=-1))
                        valid_log_var[key].append(stat_func(log_var.cpu().detach().numpy(), axis=-1))

                    
                if lr_scheduler is not None:
                    lr_scheduler.step(np.mean(valid_total_loss_with_final_kl))

                end_time = time.time()
                print('%d/%d' % (record_i, times_per_epoch_to_record), end = ' - ')
                print('TRAIN:: ', end='', file=sys.stderr)
                print('rec loss: %.7f' % np.mean(train_rec_loss), end=' -- ')
                print('kl-div: %.7f' % np.mean(train_kl), end=' - ')
                print('total loss: %.7f' % np.mean(train_total_loss), end=' - ')
                print('Loss: %.7f' % np.mean(train_total_loss_with_final_kl), end=' - ')
                print('VALID:: ', end='')
                print('rec loss: %.7f' % np.mean(valid_rec_loss), end=' - ')
                print('kl-div: %.7f' % np.mean(valid_kl), end=' - ')
                print('total loss: %.7f' % np.mean(valid_total_loss), end=' - ')
                print('Loss: %.7f' % np.mean(valid_total_loss_with_final_kl), end=' - ')
                print('Time (s): %.1f' % (end_time - start_time))
                sys.stdout.flush()


                # record best model on validation rec loss
                    
                if np.mean(valid_rec_loss) < lowest_rec_loss:
                    lowest_rec_loss = deepcopy(np.mean(valid_rec_loss))
                    lowest_rec_loss_kl = deepcopy(np.mean(valid_kl))
                    torch.save(model.state_dict(), os.path.join(experiment_dir, 'lowest_rec_loss_model.pt'))
                
                if np.mean(valid_total_loss_with_final_kl) < lowest_total_loss_with_final_kl:
                    lowest_total_loss_with_final_kl = deepcopy(np.mean(valid_total_loss_with_final_kl))
                    lowest_total_loss_with_final_kl_kl = deepcopy(np.mean(valid_kl))
                    torch.save(model.state_dict(), os.path.join(experiment_dir, 'lowest_total_loss_with_final_kl_model.pt'))
                

                record_i += 1
                global_record_i += 1

                train_sf_rec_loss, train_rec_loss, train_kl, train_sf_reg, train_total_loss, train_total_loss_with_final_kl = [], [], [], [], [], []
                train_mean, train_log_var, train_sf, train_sf_rec = {'Mean': [], 'Min': [], 'Max': []}, {'Mean': [], 'Min': [], 'Max': []}, {'Mean': [], 'Min': [], 'Max': []}, {'Mean': [], 'Min': [], 'Max': []}
                start_time = time.time()

    # record final model (more regularized than reported best model)
    # torch.save(cgvae.state_dict(), os.path.join(local_experiment_dir, 'final_model.pt'))
    
    # record hyperparameters and final best metrics
    metrics_dict = {'lowest_rec_loss': lowest_rec_loss,
                    'kl_at_lowest_rec_loss': lowest_rec_loss_kl,
                    'final_rec_loss': np.mean(valid_rec_loss),
                    'final_kld': np.mean(valid_kl),
                    'lowest_total_loss_with_final_kl': lowest_total_loss_with_final_kl,
                    'lowest_total_loss_with_final_kl_kl': lowest_total_loss_with_final_kl_kl
                    }
    
    with open(os.path.join(experiment_dir, 'validation_metrics.json'), 'w+') as f:
        json.dump(metrics_dict, f, indent=2)
    

