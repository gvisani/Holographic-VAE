
import os, sys

import gzip, pickle
import json
import numpy as np

import scipy

import argparse

from experiments.ATOM3D_LBA.src.training import hvae_inference_atom3d_lba, select_best_hparams, get_regressor

from experiments.ATOM3D_LBA.src.utils import get_pdb_to_neglogkdki_by_split

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--output_filename', type=str, default='ATOM3D_latent_space.gz')
    parser.add_argument('--pdb_similarity_split', type=int, default=30, choices=[30, 60])
    parser.add_argument('--regressor', type=str, default='RF', choices=['linear_sklearn', 'ridge', 'linear_pytorch', 'RF', 'XGBoost', 'MLP'])
    parser.add_argument('--aggr_fn', type=str, default='sum', choices=['sum', 'mean'])
    parser.add_argument('--num_bootstrap_samples', type=int, default=10)
    parser.add_argument('--seed', type=int, default=123456789)


    args = parser.parse_args()

    path_to_raw_data = os.path.join(args.data_path, 'LBA-raw/pdbbind_2019-refined-set/data/')
    path_to_indices = os.path.join(args.data_path, f'LBA-split-by-sequence-identity-{args.pdb_similarity_split}-indices')

    if not os.path.exists(os.path.join(args.model_dir, args.output_filename)):
        '''
        On a single A40 GPU, with single CPU, batch_size=1, this takes ~ minutes
        '''
        import time
        start = time.time()
        hvae_inference_atom3d_lba(args.model_dir, path_to_raw_data, output_filename=args.output_filename)
        end = time.time()
        print('Inference took %f seconds.' % (end-start))
        print()

    with gzip.open(os.path.join(args.model_dir, args.output_filename), 'rb') as f:
        results = pickle.load(f)

    pdb_to_neglogkdki_by_split = get_pdb_to_neglogkdki_by_split(path_to_raw_data, path_to_indices)

    if args.aggr_fn == 'mean':
        aggr_fun = np.mean
    elif args.aggr_fn == 'sum':
        aggr_fun = np.sum
    else:
        raise ValueError('Invalid aggregation function. Must be either "mean" or "sum".')
    
    X_by_split = {}
    y_by_split = {}
    for split in ['train', 'val', 'test']:
        X_by_split[split] = []
        y_by_split[split] = []
        for pdb, inv in zip(results['pdbs'], results['invariants']):
            if pdb in pdb_to_neglogkdki_by_split[split]:
                X_by_split[split].append(aggr_fun(inv, axis=0))
                y_by_split[split].append(pdb_to_neglogkdki_by_split[split][pdb])
        X_by_split[split] = np.vstack(X_by_split[split])
        y_by_split[split] = np.array(y_by_split[split])
    
        print(f'X_{split}.shape = {X_by_split[split].shape}')
        print(f'y_{split}.shape = {y_by_split[split].shape}')
    print()
    
    print('Selecting best hyperparameters...')
    best_hparams_dict = select_best_hparams(args, X_by_split['train'], y_by_split['train'], X_by_split['val'], y_by_split['val'], verbose=True)
    print('Best hyperparameters: ', end='')
    print(best_hparams_dict['RMSD'])
    print()

    regressor = get_regressor(args.regressor, best_hparams_dict['RMSD'])

    predictions = {
        'train': [],
        'val': [],
        'test': []
    }

    results = {
        'train': [],
        'val': [],
        'test': []
    }

    for i in range(args.num_bootstrap_samples):
        print(f'---------- {i+1}/{args.num_bootstrap_samples} ----------')
        print()

        bootstrap_idxs = np.random.default_rng(args.seed).choice(np.arange(X_by_split['train'].shape[0]), size=X_by_split['train'].shape[0], replace=True)

        regressor.fit(X_by_split['train'][bootstrap_idxs], y_by_split['train'][bootstrap_idxs])


        for split in ['train', 'val', 'test']:

            print(f'----- {split} -----')

            y_pred = regressor.predict(X_by_split[split])

            # compute RMSE
            print('RMSE = %.3f' % (np.sqrt(np.mean((y_pred - y_by_split[split])**2))))
            
            # compute pearson correlation
            print('Pearson r: %.3f' % (scipy.stats.pearsonr(y_pred, y_by_split[split])[0]))

            print()

            predictions[split].append(y_pred)
    
    print('----- Averaged results -----')
    for split in ['train', 'val', 'test']:
        rmsd, pearson = [], []
        for i in range(10):
            rmsd.append(np.sqrt(np.mean((predictions[split][i] - y_by_split[split])**2)))
            pearson.append(scipy.stats.pearsonr(predictions[split][i], y_by_split[split])[0])
        print(f'----- {split} -----')
        print('RMSE = %.3f +/- %.3f' % (np.mean(rmsd), np.std(rmsd)))
        print('Pearson r: %.3f +/- %.3f' % (np.mean(pearson), np.std(pearson)))
    print()
        
    print('----- Bootstrapped results -----')
    for split in ['train', 'val', 'test']:
        predictions[split] = np.mean(np.vstack(predictions[split]), axis=0)
        print(f'----- {split} -----')
        print('RMSE = %.3f' % (np.sqrt(np.mean((predictions[split] - y_by_split[split])**2))))
        print('Pearson r: %.3f' % (scipy.stats.pearsonr(predictions[split], y_by_split[split])[0]))
        print()
    

        


