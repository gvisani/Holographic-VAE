



import numpy as np

import h5py
import hdf5plugin

from tqdm import tqdm


def add_noise(np_protein: np.ndarray, noise_level: float):
    '''
    Adds noise to protein coordinates, sampled from a gaussian distribution with mean 0 and std noise_level.

    np_protein: np.ndarray
        Protein data, as a numpy array, as outputted by get_structural_info() routine.
    noise_level: float
        Standard deviation - in angstroms - of the gaussian distribution from which the noise is sampled.
    '''

    real_idxs = np_protein['res_ids'][:, 0] != b''

    coords = np_protein['coords']
    noise = np.random.normal(0, noise_level, (np.sum(real_idxs), 3))

    coords[real_idxs] += noise

    np_protein['coords'] = coords

    return np_protein





if __name__ == '__main__':

    datapath = '/gscratch/stf/gvisan01/casp12/structural_info/casp12_testing_structural_info.hdf5'

    noise_level = 0.3

    with h5py.File(datapath, 'r') as f:
        np_proteins = f['data']

        for i in tqdm(range(len(np_proteins))):
            np_protein = np_proteins[i]
            print(np_protein['coords'])
            np_protein = add_noise(np_protein, noise_level)
            print(np_protein['coords'])
            break

            # np_proteins[i] = np_protein

