

import os, sys
from get_structural_info import get_structural_info_from_protein, pad_structural_info
from preprocessor_pdbs import PDBPreprocessor
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import h5py
import hdf5plugin
from hdf5plugin import LZ4
import sys
from progress.bar import Bar

def callback(pdb_file, parser):

    try:
        pdb,ragged_structural_info = get_structural_info_from_protein(pdb_file, parser)
        mat_structural_info = pad_structural_info(
            ragged_structural_info,padded_length=200000
        )
    except Exception as e:
        print(e)
        print('Error with', )
        return (None,)

    
    return (pdb, *mat_structural_info)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--output_hdf5', type=str, required=True,
                        help='User-defined name of output hdf5 file that will contain the per-pdb structural information.')

    parser.add_argument('--output_dataset_name', type=str, default='data',
                        help='Name of the dataset within output_hdf5 where the structural information will be saved. We recommend keeping this set to simply "data".')
    
    parser.add_argument('--parallelism', type=int, default=1,
                        help='Parallelism for multiprocessing.')

    parser.add_argument('--pdb_list', type=str, required=True,
                        help='csv file containing list of PDB files of interest, under the column "pdb".')

    parser.add_argument('--pdb_dir', type=str, required=True,
                        help='Directory containing PDB files.')
    
    parser.add_argument('--parser', type=str, default='biopython', choices=['biopython', 'pyrosetta'],
                        help='Which PDB parser to use.')
    
    args = parser.parse_args()

    pdb_list = list(pd.read_csv(args.pdb_list)['pdb'])

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
        ('pdb','S50',()),
        ('atom_names', 'S4', (max_atoms)),
        ('elements', 'S1', (max_atoms)),
        ('res_ids', 'S6', (max_atoms, 6)),
        ('coords', 'f4', (max_atoms, 3)),
        # ('SASAs', 'f4', (max_atoms)), # unused in H-(V)AE
        # ('charges', 'f4', (max_atoms)), # unused in H-(V)AE
    ])
    with h5py.File(args.output_hdf5,'w') as f:
        f.create_dataset(args.output_dataset_name,
                         shape=(ds.size,),
                         dtype=dt,
                         compression=LZ4())
    with Bar('Processing', max = ds.count(), suffix='%(percent).1f%%') as bar:
        with h5py.File(args.output_hdf5,'r+') as f:
            for i,structural_info in enumerate(ds.execute(
                    callback,
                    limit = None,
                    params = {},
                    parallelism = args.parallelism)):
                if structural_info[0] is None:
                    bar.next()
                    n+=1
                    continue
                pdb,atom_names,elements,res_ids,coords = (*structural_info,) #,sasas,charges
                #print(pdb)
                #print(max_atoms - np.sum(atom_names == b''),'atoms in array')
                #print('wrting to hdf5')
                try:
                    f[args.output_dataset_name][i] = (pdb,atom_names,elements,res_ids,coords) # ,sasas,charges)
                    #print('success')
                except Exception as e:
                    print(e)
                #print('done writing. \n moving to next entry')
                n+=1
                bar.next()
    
