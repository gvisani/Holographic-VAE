
import os, sys
import gzip, pickle
import json

import numpy as np
import torch
import torch.nn.functional as F
from e3nn import o3

from holographic_vae.models import H_VAE
from holographic_vae.so3.functional import put_dict_on_device, make_vec
from holographic_vae.cg_coefficients import get_w3j_coefficients
from holographic_vae.utils.loss_functions import *

from typing import *


def generate_dataset_of_conditional_samples(experiment_dir: str,
                                            num_samples_per_label: int = 1000, # 1000 samples per label in the original MNIST dataset
                                            model_name: str = 'lowest_total_loss_with_final_kl_model',
                                            seed: int = 123456789):
        
    # get hparams from json
    with open(os.path.join(experiment_dir, 'hparams.json'), 'r') as f:
        hparams = json.load(f)
    
    data_irreps = o3.Irreps.spherical_harmonics(hparams['lmax'], 1)

    # setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Running on %s.' % (device))

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
    
    # create model and load weights
    model = H_VAE(data_irreps, w3j_matrices, hparams['model_hparams'], device, normalize_input_at_runtime=False).to(device)
    model.load_state_dict(torch.load(os.path.join(experiment_dir, model_name + '.pt'), map_location=torch.device(device)))
    model.eval()

    num_params = 0
    for param in model.parameters():
        num_params += torch.flatten(param.data).shape[0]
    print('There are %d parameters' % (num_params))
    sys.stdout.flush()


    ## initialize generator
    generator = torch.Generator().manual_seed(seed)

    sampled_projections = []
    sampled_frames = []
    sampled_labels = []
    for label in range(10):
    
        z = torch.normal(torch.zeros((num_samples_per_label, hparams['model_hparams']['latent_dim'])), torch.ones((num_samples_per_label, hparams['model_hparams']['latent_dim'])), generator=generator)

        frame = torch.eye(3).repeat(num_samples_per_label, 1).float().view(-1, 3, 3).squeeze().to(device)

        y = torch.full((num_samples_per_label,), label).long()

        z = model.condition_latent_space(z.to(device), F.one_hot(y.long(), num_classes=10).float().to(device))

        projection = make_vec(model.decode(z.to(device), frame.to(device)))

        sampled_projections.append(projection.detach().cpu())
        sampled_frames.append(frame.detach().cpu())
        sampled_labels.append(y.detach().cpu())
    
    sampled_projections = torch.cat(sampled_projections, dim=0)
    sampled_frames = torch.cat(sampled_frames, dim=0)
    sampled_labels = torch.cat(sampled_labels, dim=0)

    # save data
    data_dict = {
        'sampled_test': {
            'projections': sampled_projections,
            'labels': sampled_labels,
            'rotations': sampled_frames
        }
    }
    with gzip.open(os.path.join(experiment_dir, 'sampled_data_for_testing_classifier.pkl.gz'), 'wb') as f:
        pickle.dump(data_dict, f)


if __name__ == '__main__':
    generate_dataset_of_conditional_samples(experiment_dir=sys.argv[1])


