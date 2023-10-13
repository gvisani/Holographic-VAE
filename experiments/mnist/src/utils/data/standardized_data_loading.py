
import os
import gzip, pickle

from e3nn import o3

from experiments.mnist.src.utils.data import SphericalMNISTDataset


def load_data(hparams, splits=['train', 'valid', 'test'], test_filepath=None):
    '''
    "test_filepath" overrides the default choice of splits
    '''

    data_irreps = o3.Irreps.spherical_harmonics(hparams['lmax'], 1)

    if test_filepath is None:

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
    
    else:
        assert len(splits) == 1, 'If "test_filepath" is specified, then only one split can be specified, which becomes the name of the dataset.'
        dataset_name = splits[0]

        with gzip.open(test_filepath, 'rb') as f:
            data_dict = pickle.load(f)
        
        datasets = {}
        datasets[dataset_name] = SphericalMNISTDataset(data_dict[dataset_name]['projections'], data_irreps, data_dict[dataset_name]['labels'], data_dict[dataset_name]['rotations'])

    return datasets, data_irreps
