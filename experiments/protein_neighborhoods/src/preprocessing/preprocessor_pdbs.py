import pandas as pd
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

def process_data_dir(pdb,pdb_dir):
    assert(process_data_dir.callback)

    pdb = pdb if isinstance(pdb, str) else pdb.decode('utf-8')

    pdb_file = os.path.join(pdb_dir, pdb + '.pdb')

    #print('pdb is ',pdb,pose.pdb_info().name())
    return process_data_dir.callback(pdb_file, **process_data_dir.params)

def initializer(init, callback, params, init_params):
    if init is not None:
        init(**init_params)
    process_data_dir.callback = callback
    process_data_dir.params = params
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class PDBPreprocessor:
    def __init__(self, pdb_list, pdb_dir):
        self.pdb_dir = pdb_dir
        self.__data = pdb_list
        self.size = len(pdb_list)
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
            process_data = functools.partial(process_data_dir,pdb_dir=self.pdb_dir)
            for res in pool.imap(process_data, data):
                if res:
                    yield res

