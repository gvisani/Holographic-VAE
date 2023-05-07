import signal
import numpy as np
import time
import os
import logging
import itertools
import functools
import warnings
from multiprocessing import Pool, TimeoutError
import h5py
import sys




def process_data(ind, hdf5_file, key):
    assert(process_data.callback)
    with h5py.File(hdf5_file, 'r') as f:
        protein = f[key][ind]
    return process_data.callback(protein, **process_data.params)

def initializer(init, callback, params, init_params):
    if init is not None:
        init(**init_params)
    process_data.callback = callback
    process_data.params = params
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class PDBPreprocessor:
    def __init__(self, hdf5_file, key):

        with h5py.File(hdf5_file,'r') as f:
            num_proteins = f[key].shape[0]

        self.key = key
        self.hdf5_file = hdf5_file
        self.size = num_proteins
        self.__data = np.arange(num_proteins)

        print(self.size)
        print(self.hdf5_file)
        
    def count(self):
        return len(self.__data)

    def execute(self, callback, parallelism = 8, limit = None, params = None, init = None, init_params = None):
        if limit is None:
            data = self.__data
        else:
            data = self.__data[:limit]
        with Pool(initializer = initializer, processes=parallelism, initargs = (init, callback, params, init_params)) as pool:
    
            all_loaded = True
            if all_loaded:
                # logging.info('All PDB files are loaded.')
                pass
            else:
                raise Exception("Some PDB files could not be loaded.")
            
            process_data_hdf5 = functools.partial(
                process_data,
                hdf5_file=self.hdf5_file,
                key=self.key
            )
            ntasks = data.shape[0]
            num_cpus = os.cpu_count()
            chunksize = ntasks // num_cpus + 1
            print('Data size = {}, cpus = {}, chunksize = {}'.format(ntasks,num_cpus,chunksize))

            if chunksize > 8:
                chunksize = 8
            for res in pool.imap(process_data_hdf5, data, chunksize=chunksize):
                if res:
                    yield res

