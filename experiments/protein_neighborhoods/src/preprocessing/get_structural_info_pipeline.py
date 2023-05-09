

import os, sys
from get_structural_info import get_structural_info_from_protein, pad_structural_info
from preprocessor_pdbs import PDBPreprocessor
from argparse import ArgumentParser
import numpy as np
import h5py
import sys
import logging
from progress.bar import Bar

def callback(pose):

    if pose is None:
        print('pose is none')
        return (None,)

    try:
        pdb,ragged_structural_info = get_structural_info_from_protein(pose)
        mat_structural_info = pad_structural_info(
            ragged_structural_info,padded_length=200000
        )
    except Exception as e:
        print(e)
        print('Error with',pose.pdb_info().name())
        return (None,)

    
    return (pdb, *mat_structural_info)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--hdf5_out', dest='hdf5_out', type=str,
                        help='Output hdf5 filename')

    parser.add_argument('--dataset_name', dest='dataset_name', type=str, default='data',
                        help='Chosen name of dataset within hdf5_out. I put just some dummy name')

    parser.add_argument('--parallelism', dest='parallelism', type=int, default=4,
                        help='How many processed to run in parallel (make it equal to number of tasks requested in hyak)')

    parser.add_argument('--pdb_filepath', dest='pdb_filepath', type=str, default=None,
                        help='path to a .txt file containing pdbs to collect, one per line')

    parser.add_argument('--pdb_dir', dest='pdb_dir', type=str, default=None,
                        help='directory of pdb files')
    
    args = parser.parse_args()

    pdb_list = []
    with open(args.pdb_filepath) as f:
        for line in f:
            pdb_list.append(line.strip())

    pdb_list_from_dir = []
    for file in os.listdir(args.pdb_dir):
        if file.endswith(".pdb"):
            pdb = file[:4]
            pdb_list_from_dir.append(pdb)
    
    # filter out pdbs that are not in the directory
    pdb_list = list(set(pdb_list) & set(pdb_list_from_dir))
    
    ds = PDBPreprocessor(pdb_list, args.pdb_dir)
    bad_neighborhoods = []
    n = 0

    max_atoms = 200000
    dt = np.dtype([
        ('pdb','S4',()),
        ('atom_names', 'S4', (max_atoms)),
        ('elements', 'S1', (max_atoms)),
        ('res_ids', 'S6', (max_atoms, 6)),
        ('coords', 'f4', (max_atoms, 3)),
        ('SASAs', 'f4', (max_atoms)),
        # ('RSAs', 'f4', (max_atoms)),
        ('charges', 'f4', (max_atoms)),
    ])
    with h5py.File(args.hdf5_out,'w') as f:
        f.create_dataset(args.dataset_name,
                         shape=(ds.size,),
                         dtype=dt)
    with Bar('Processing', max = ds.count(), suffix='%(percent).1f%%') as bar:
        with h5py.File(args.hdf5_out,'r+') as f:
            for i,structural_info in enumerate(ds.execute(
                    callback,
                    limit = None,
                    params = {},
                    parallelism = args.parallelism)):
                if structural_info[0] is None:
                    bar.next()
                    n+=1
                    continue
                pdb,atom_names,elements,res_ids,coords,sasas,charges = (*structural_info,) #,rsas
                #print(pdb)
                #print(max_atoms - np.sum(atom_names == b''),'atoms in array')
                #print('wrting to hdf5')
                try:
                    f[args.dataset_name][i] = (pdb,atom_names,elements,res_ids,coords,sasas,charges) # ,sasas,rsas,charges)
                    #print('success')
                except Exception as e:
                    print(e)
                #print('done writing. \n moving to next entry')
                n+=1
                bar.next()
    
