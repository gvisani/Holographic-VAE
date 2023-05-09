
from turtle import back
from get_neighborhoods import get_neighborhoods_from_protein, pad_neighborhoods
from preprocessor_hdf5_proteins import PDBPreprocessor
from argparse import ArgumentParser
import numpy as np
import h5py
import sys
import logging
from progress.bar import Bar
import traceback
from protein_holography_pytorch.utils.argparse import *

MAX_ATOMS = 1000
ALL_AAs = ['R', 'H', 'K', 'D', 'E', 'S', 'T', 'N', 'Q', 'C', 'U', 'G', 'P', 'A', 'I', 'L', 'M', 'F', 'W', 'Y', 'V']

# Modify the two paths below with your local paths, if you have downloaded proteinnet.
PROTEINNET_VALIDATION_CHAINS = '/gscratch/scrubbed/mpun/test/casp12/validation'
PROTEINNET_TRAINING_CHAINS = '/gscratch/scrubbed/mpun/test/casp12/training_30'

def get_proteinnet__pdb_chain_pairs(testing=False):
    if testing:
        f = open(PROTEINNET_VALIDATION_CHAINS)
        print('Using ProteinNet validation chains.')
    else:
        f = open(PROTEINNET_TRAINING_CHAINS)
        print('Using ProteinNet training_30 chains.')
    lines = f.readlines()
    f.close()

    pdbs = []
    chains = []
    d_pdbs = []
    id_line = False
    for line in lines[:]:
        if id_line:
            id_line = False
            split_line = line.split('_')
            pdb = split_line[0]
            if testing:
                pdb = pdb.split('#')[1]
            pdbs.append(pdb)
            if len(split_line) == 3:
                chains.append(split_line[2].split('\n')[0])
            else:
                chains.append(split_line[1][-3].upper())
                d_pdbs.append(split_line[0])
        if "[ID]" in line:
            id_line=True
    
    return set(map(lambda x: '_'.join(x), zip(pdbs, chains)))

def callback(np_protein, r, remove_central_residue, backbone_only):

    try:
        neighborhoods = get_neighborhoods_from_protein(np_protein, r=r, remove_central_residue=remove_central_residue, backbone_only=backbone_only)
        padded_neighborhoods = pad_neighborhoods(neighborhoods, padded_length=1000)
    except Exception as e:
        print(e)
        print('Error with ', np_protein[0])
        return (None,)
    
    return (padded_neighborhoods)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--input_hdf5', type=str, required=True,
                        help='Path to hdf5 file containing protein 3D structure information. Must be output to the script `get_structural_info_pipeline.py`')
    parser.add_argument('--output_hdf5', type=str, required=True,
                        help='User-defined name of output hdf5 file that will contain the extracted neighborhoods.')
    parser.add_argument('--input_dataset_name', type=str, default='data',
                        help='Name of the dataset within input_hdf5 where the structural information is to be found. We recommend keeping this set to simply "data".')
    parser.add_argument('--output_dataset_name', type=str, default='data',
                        help='Name of the dataset within output_hdf5 where the neighborhoods information will be saved. We recommend keeping this set to simply "data".')
    parser.add_argument('--parallelism', type=int, default=1,
                        help='Parallelism count for multiprocessing')
    parser.add_argument('--radius', type=float, default=10.0,
                        help='Radius of the neighborhoods. Alias of "rcut".')
    parser.add_argument('--remove_central_residue', type=str_to_bool, default=False,
                        help='Whether to remove the central residue. Set it to False for H-(V)AE neighborhoods.')
    parser.add_argument('--backbone_only', type=str_to_bool, default=False,
                        help='Whether to keep only backbone atoms. Set it to False for H-(V)AE neighborhoods.')
    parser.add_argument('--filter_out_chains_not_in_proteinnet', type=str_to_bool, default=False,
                        help='Whether to exclude neighborhoods that do not appear in casp12 proteinnet, as only some chains within the provided PDBs are in proteinnet. \
                              Set to True in our experiments on CASP 12, but it requires access to the proteinnet data.')
    parser.add_argument('--AAs', type=str, default='all',
                        help='List of amino-acid types to collect. Either "all" or provided in comma-separated form.')
    
    args = parser.parse_args()

    if args.AAs == 'all':
        filter_AAs = set(ALL_AAs)
    else:
        filter_AAs = set(args.AAs.split(','))
    
    if args.filter_out_chains_not_in_proteinnet:
        print('Filtering out chains not in ProteinNet.')
        try:
            proteinnet__pdb_chain_pairs = get_proteinnet__pdb_chain_pairs(testing=True if 'testing' in args.input_hdf5 else False)
        except FileNotFoundError:
            print('Could not find ProteinNet file. Ignoring.')
    
        
    logging.basicConfig(level=logging.DEBUG)
    ds = PDBPreprocessor(args.input_hdf5, args.input_dataset_name)

    n_not_proteinnet = 0
    dt = np.dtype([
        ('res_id','S50', (6)),
        ('atom_names', 'S4', (MAX_ATOMS)),
        ('elements', 'S1', (MAX_ATOMS)),
        ('res_ids', 'S50', (MAX_ATOMS, 6)),
        ('coords', 'f4', (MAX_ATOMS, 3)),
        ('SASAs', 'f4', (MAX_ATOMS)),
        ('charges', 'f4', (MAX_ATOMS)),
    ])
    print(dt)
    print('writing hdf5 file')
    curr_size = 1000
    with h5py.File(args.output_hdf5, 'w') as f:
        # Initialize dataset
        f.create_dataset(args.output_dataset_name,
                         shape=(curr_size,),
                         maxshape=(None,),
                         dtype=dt)
        
    print('calling parallel process')
    with Bar('Processing', max = ds.count(), suffix='%(percent).1f%%') as bar:
        with h5py.File(args.output_hdf5, 'r+') as f:
            n = 0
            for i, neighborhoods in enumerate(ds.execute(callback,
                                                         limit = ds.size,
                                                         params = {'r': args.radius,
                                                                   'remove_central_residue': args.remove_central_residue,
                                                                   'backbone_only': args.backbone_only},
                                                         parallelism = args.parallelism)):
                
                if neighborhoods[0] is None:
                    bar.next()
                    continue
                
                for neighborhood in neighborhoods:

                    if args.filter_out_chains_not_in_proteinnet:
                        if '_'.join([neighborhood[0][1].decode('utf-8'), neighborhood[0][2].decode('utf-8')]) not in proteinnet__pdb_chain_pairs:
                            bar.next()
                            n_not_proteinnet += 1
                            continue

                    if n == curr_size:
                        curr_size += 1000
                        f[args.output_dataset_name].resize((curr_size,))
                    
                    # only add neighborhoods of desired AA types
                    if neighborhood[0][0].decode('utf-8') in filter_AAs:
                        f[args.output_dataset_name][n] = (*neighborhood,)
                        n += 1
                
                bar.next()

            # finally, resize dataset to be of needed shape to exactly contain the data and nothing more
            f[args.output_dataset_name].resize((n,))
        
    print('%d total neighborhoods saved.' % (n))
    if args.filter_out_chains_not_in_proteinnet:
        print('%d neighborhoods not in ProteinNet.' % (n_not_proteinnet))
    
    print('Done with parallel computing')
