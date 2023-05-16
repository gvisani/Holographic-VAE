

import os, sys

import json
import numpy as np
import scipy

import sklearn
import sklearn.linear_model
import sklearn.ensemble

from xgboost import XGBRegressor
from .predictors import LinearRegressor, MLPRegressor



def select_best_hparams(args, X_train, y_train, X_val, y_val, verbose=False):

    best_hparams_filepath = os.path.join(args.model_dir, 'ATOM3D_LBA_best_hparams-aggr_fn=%s-regressor=%s-pdb_similarity_split=%s.json' % (args.aggr_fn, args.regressor, args.pdb_similarity_split))
        
    # load best hparams, if they exist
    if os.path.exists(best_hparams_filepath):
        print('Loading best hparams from file...')
        with open(best_hparams_filepath, 'r') as f:
            best_hparams = json.load(f)
        return best_hparams
        
    print('Selecting best hparams...')
    sys.stdout.flush()

    from sklearn.model_selection import ParameterGrid

    if args.regressor == 'linear_sklearn':
        model_obj = sklearn.linear_model.LinearRegression
        hyperparams_grid = None
    elif args.regressor == 'ridge':
        model_obj = sklearn.linear_model.Ridge
        hyperparams_grid = {'alpha': [0.0001, 0.001, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]}
    elif args.regressor == 'linear_pytorch':
        model_obj = LinearRegressor
        hyperparams_grid = None
    elif args.regressor == 'RF':
        model_obj = sklearn.ensemble.RandomForestRegressor
        hyperparams_grid = {'max_features': [1.0, 0.333, 'sqrt'], 'min_samples_leaf': [2, 5, 10, 15], 'n_estimators': [32, 64, 100, 200]}
    elif args.regressor == 'XGBoost':
        model_obj = XGBRegressor
        hyperparams_grid = None
    elif args.regressor == 'MLP':
        model_obj = MLPRegressor
        hyperparams_grid = None
    else:
        raise NotImplementedError('"{}" regressor not implemented.')

    # return none if we don't want to do hyperparameter tuning with the selcted model
    if hyperparams_grid is None:
        return None

    hparams_combos = list(ParameterGrid(hyperparams_grid))

    rmsd_list = []
    pearsonr_list = []
    for hparams in hparams_combos:

        temp_rmsd_list = []
        temp_pearsonr_list = []
        for _ in range(5): # average results over 5 bootstrapped runs to get a more robust estimate of the performance
            model = model_obj(**hparams)

            bootstrapped_idxs = np.random.default_rng(args.seed).choice(np.arange(X_train.shape[0]), size=X_train.shape[0], replace=True)
            model.fit(X_train[bootstrapped_idxs], y_train[bootstrapped_idxs])

            y_hat_val = model.predict(X_val)

            temp_rmsd = np.sqrt(sklearn.metrics.mean_squared_error(y_val, y_hat_val))
            temp_pearsonr = scipy.stats.pearsonr(y_val, y_hat_val)[0]

            temp_rmsd_list.append(temp_rmsd)
            temp_pearsonr_list.append(temp_pearsonr)
        
        rmsd = np.mean(temp_rmsd_list)
        pearson = np.mean(temp_pearsonr_list)

        if verbose:
            print('RMSD: %.4f' % rmsd)
            print('Pearson r: %.4f' % pearson)

        rmsd_list.append(rmsd)
        pearsonr_list.append(pearson)

    best_hparams_by_rmsd = hparams_combos[np.argmin(rmsd_list)]
    best_hparams_by_pearsonr = hparams_combos[np.argmax(pearsonr_list)]
    best_hparams = {
        'RMSD': best_hparams_by_rmsd,
        'Pearson r':  best_hparams_by_pearsonr
    }

    # save best hparams
    os.makedirs(os.path.dirname(best_hparams_filepath), exist_ok=True)
    with open(best_hparams_filepath, 'w') as f:
        json.dump(best_hparams, f, indent=4)

    return best_hparams

def get_regressor(regressor, best_hparams):

    if regressor == 'linear_sklearn':
        model = sklearn.linear_model.LinearRegression(**best_hparams)
    elif regressor == 'ridge':
        model = sklearn.linear_model.Ridge(**best_hparams)
    elif regressor == 'linear_pytorch':
        model = LinearRegressor(**best_hparams)
    elif regressor == 'RF':
        model = sklearn.ensemble.RandomForestRegressor(**best_hparams)
    elif regressor == 'XGBoost':
        model = XGBRegressor(**best_hparams)
    elif regressor == 'MLP':
        model = MLPRegressor(**best_hparams)
    else:
        raise NotImplementedError('"{}" regressor not implemented.')
    
    return model
