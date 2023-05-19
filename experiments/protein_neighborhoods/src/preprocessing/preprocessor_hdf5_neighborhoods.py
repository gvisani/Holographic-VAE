
import signal
import numpy as np
import time
import os, sys
import logging
import itertools
import functools
import warnings
from multiprocessing import Pool, TimeoutError
import h5py
import torch

from get_zernikegrams import extract_neighborhood_info

BACKBONE_ATOMS = [b' N  ', b' CA ', b' C  ', b' O  ']
BACKBONE_ATOMS_PLUS_CB = [b' N  ', b' CA ', b' C  ', b' O  ', b' CB ']

# def process_data(nb, channels=None, backbone_only=False, request_frame=False, get_psysicochemical_info_for_hydrogens=True):
def process_data(ind, hdf5_file, hdf5_key, channels=None, backbone_only=False, request_frame=False, get_psysicochemical_info_for_hydrogens=True):
    assert (process_data.callback)

    # [replaces nothing]
    with h5py.File(hdf5_file,'r') as f:
        nb = f[hdf5_key][ind]

    selected_masks, selected_weights, frame = extract_neighborhood_info(nb, channels=channels, backbone_only=backbone_only, request_frame=request_frame, get_psysicochemical_info_for_hydrogens=get_psysicochemical_info_for_hydrogens)
    
    return process_data.callback(nb, selected_masks, selected_weights, frame, **process_data.params)


def initializer(init, callback, params, init_params):
    if init is not None:
        init(**init_params)
    process_data.callback = callback
    process_data.params = params
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class HDF5Preprocessor:
    def __init__(self, hdf5_file, hdf5_key):
        with h5py.File(hdf5_file, 'r') as f:
            # data = np.unique(np.array(f[hdf5_key]), axis=0)
            num_neighborhoods = np.array(f[hdf5_key].shape[0])

        # self.__data = data
        self.__data = np.arange(num_neighborhoods)
        self.hdf5_file = hdf5_file
        self.hdf5_key = hdf5_key

        self.size = self.__data.shape[0]

    def count(self):
        return self.__data.shape[0]

    def execute(self, callback, channels = None, backbone_only = False, request_frame = False, get_psysicochemical_info_for_hydrogens = True, parallelism = None, limit = None, params = None, init = None, init_params = None):
        if limit is None:
            data = self.__data
        else:
            data = self.__data[:limit]
        with Pool(initializer = initializer, processes=parallelism, initargs = (init, callback, params, init_params)) as pool:

            process_data_hdf5 = functools.partial(
                process_data,
                hdf5_file=self.hdf5_file,
                hdf5_key=self.hdf5_key,
                channels=channels,
                backbone_only=backbone_only,
                request_frame=request_frame,
                get_psysicochemical_info_for_hydrogens=get_psysicochemical_info_for_hydrogens
            )

            # ntasks = self.size
            # num_cpus = os.cpu_count()
            # chunksize = ntasks // num_cpus + 1

            # print('Parallelism: ', parallelism, file=sys.stderr)
            # print('Number of tasks: ', ntasks, file=sys.stderr)
            # print('Number of cpus: ', num_cpus, file=sys.stderr)
            # print('Chunksize: ', chunksize, file=sys.stderr)

            # if chunksize > 100:
            #     chunksize = 16
            
            # print('Chunksize: ', chunksize, file=sys.stderr)

            for coords in pool.imap(process_data_hdf5, data): #, chunksize=chunksize):
                if coords:
                    yield coords

