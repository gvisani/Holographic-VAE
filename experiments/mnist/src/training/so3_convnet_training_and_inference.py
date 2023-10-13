
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
from sklearn.metrics import accuracy_score, classification_report
from scipy.special import softmax
from copy import deepcopy

from typing import *

from holographic_vae.models import SO3_ConvNet_InvariantOutput
from holographic_vae.so3.functional import put_dict_on_device
from holographic_vae.cg_coefficients import get_w3j_coefficients




def so3_convnet_training(experiment_dir: str):
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
    model = SO3_ConvNet_InvariantOutput(data_irreps, w3j_matrices, hparams['model_hparams'], normalize_input_at_runtime=False).to(device)
    
    num_params = 0
    for param in model.parameters():
        num_params += torch.flatten(param.data).shape[0]
    print('There are %d parameters' % (num_params), flush=True)

    # setup learning algorithm
    
    if hparams['weight_decay']:
        optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'], weight_decay=1e-5)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'])

    if hparams['lr_scheduler'] is None or hparams['lr_scheduler'] == 'constant':
        lr_scheduler = None
    elif hparams['lr_scheduler'] == 'reduce_lr_on_plateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)


    lowest_valid_loss = np.inf
    times_per_epoch_to_record = 5
    steps_to_record = len(train_dataloader) // times_per_epoch_to_record
    for epoch in range(hparams['n_epochs']):
        print('Epoch %d/%d' % (epoch+1, hparams['n_epochs']), flush=True)

        record_i = 1

        train_loss_trace = []

        start_time = time.time()
        for i, (X, X_vec, y, rot) in enumerate(train_dataloader):
            X = put_dict_on_device(X, device)
            y = y.long().to(device)

            optimizer.zero_grad()
            model.train()

            y_hat_probas = model(X)
            loss = F.cross_entropy(y_hat_probas, y)
            loss.backward()
            optimizer.step()

            train_loss_trace.append(loss.item())

            if i % steps_to_record == (steps_to_record - 1):
                valid_loss_trace = []
                y_hat_probas_trace = []
                y_trace = []

                for i, (X, X_vec, y, rot) in enumerate(valid_dataloader):
                    X = put_dict_on_device(X, device)
                    y = y.long().to(device)

                    model.eval()

                    y_hat_probas = model(X)
                    loss = F.cross_entropy(y_hat_probas, y) 
                    valid_loss_trace.append(loss.item())
                    y_hat_probas_trace.append(y_hat_probas.detach().cpu())
                    y_trace.append(y.detach().cpu())

                valid_loss = np.mean(valid_loss_trace)
                valid_accuracy = accuracy_score(torch.cat(y_trace, dim=0).cpu().numpy(), torch.cat(y_hat_probas_trace, dim=0).argmax(dim=1).cpu().numpy())

                print('Step %d/%d, Train Loss: %.4f, Valid Loss: %.4f, Valid Accuracy: %.4f, Time (s): %.2f' % (record_i, times_per_epoch_to_record, np.mean(train_loss_trace), np.mean(valid_loss_trace), valid_accuracy, time.time() - start_time), flush=True)


                if valid_loss < lowest_valid_loss:
                    lowest_valid_loss = deepcopy(valid_loss)
                    torch.save(model.state_dict(), os.path.join(experiment_dir, 'lowest_valid_loss_model.pt'))
                
                train_loss_trace = []
                record_i += 1
                start_time = time.time()
                
                if lr_scheduler is not None:
                    lr_scheduler.step(valid_loss)

    # record final best metrics
    metrics_dict = {
        'lowest_valid_loss': lowest_valid_loss
    }
    with open(os.path.join(experiment_dir, 'validation_metrics.json'), 'w+') as f:
        json.dump(metrics_dict, f, indent=2)
    

    print('Training complete.', flush=True)



def so3_convnet_inference(experiment_dir: str,
                    split: str = 'test',
                    test_filepath: Optional[str] = None,
                    model_name: str = 'lowest_total_loss_with_final_kl_model',
                    verbose: bool = False,
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
    datasets, data_irreps = load_data(hparams, splits=[split], test_filepath=test_filepath)
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
    model = SO3_ConvNet_InvariantOutput(data_irreps, w3j_matrices, hparams['model_hparams'], normalize_input_at_runtime=False).to(device)
    model.load_state_dict(torch.load(os.path.join(experiment_dir, model_name + '.pt'), map_location=torch.device(device)))
    model.eval()

    num_params = 0
    for param in model.parameters():
        num_params += torch.flatten(param.data).shape[0]
    if verbose: print('There are %d parameters' % (num_params), flush=True)


    test_loss_trace = []
    y_hat_probas_trace = []
    y_trace = []
    X_vec_trace = []

    for i, (X, X_vec, y, rot) in enumerate(test_dataloader):
        X = put_dict_on_device(X, device)
        y = y.long().to(device)

        model.eval()

        y_hat_probas = model(X)
        loss = F.cross_entropy(y_hat_probas, y)
        test_loss_trace.append(loss.item())
        y_hat_probas_trace.append(y_hat_probas.detach().cpu())
        y_trace.append(y.detach().cpu())
        X_vec_trace.append(X_vec.detach().cpu())

    test_loss = np.mean(test_loss_trace)
    test_accuracy = accuracy_score(torch.cat(y_trace, dim=0).cpu().numpy(), torch.cat(y_hat_probas_trace, dim=0).argmax(dim=1).cpu().numpy())
    print('Test Loss: %.4f, Test Accuracy: %.4f' % (test_loss, test_accuracy), flush=True)
    report = classification_report(torch.cat(y_trace, dim=0).cpu().numpy(), torch.cat(y_hat_probas_trace, dim=0).argmax(dim=1).cpu().numpy())
    print(report, flush=True)

    y_trace = torch.cat(y_trace, dim=0).numpy()
    y_hat_probas_trace = torch.cat(y_hat_probas_trace, dim=0).numpy()
    X_vec_trace = torch.cat(X_vec_trace, dim=0).numpy()

    # save results
    np.savez(output_filepath,
                labels_N=y_trace,
                pred_label_probas_NC=y_hat_probas_trace,
                images_NF=X_vec_trace)



