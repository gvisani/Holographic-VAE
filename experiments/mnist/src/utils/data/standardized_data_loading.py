
import os
import gzip, pickle

from e3nn import o3

from experiments.mnist.src.utils.data import SphericalMNISTDataset


def load_data(hparams, splits=['train', 'valid', 'test']):

    data_irreps = o3.Irreps.spherical_harmonics(hparams['lmax'], 1)

    input_type_map = {
        'NRNR': '-no_rotate_train-no_rotate_test-cz=%d' % (hparams['cz']),
        'RR': '-cz=%d' % (hparams['cz']),
        'NRR': '-no_rotate_train-cz=%d' % (hparams['cz']),
        'RNR': '-no_rotate_test-cz=%d' % (hparams['cz'])
    }
    data_file = os.path.join(hparams['data_filepath'] % (input_type_map[hparams['input_type']], hparams['bw'], hparams['lmax'], hparams['normalize'], hparams['quad_weights']))

    with gzip.open(data_file, 'rb') as f:
        data_dict_all_splits = pickle.load(f)
    
    datasets = {}
    for split in splits:
        datasets[split] = SphericalMNISTDataset(data_dict_all_splits[split]['projections'], data_irreps, data_dict_all_splits[split]['labels'], data_dict_all_splits[split]['rotations'])

    return datasets, data_irreps
