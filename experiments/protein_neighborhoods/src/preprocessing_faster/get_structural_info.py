"""Module for parallel processing of pdb files into structural info"""

from argparse import ArgumentParser
import logging
import os, sys
from time import time
from typing import Tuple

import h5py
from hdf5plugin import LZ4
import numpy as np
import pandas as pd
from progress.bar import Bar

from protein_holography_pytorch.preprocessing_faster.utils.structural_info import (
    get_structural_info_from_protein__pyrosetta, pad_structural_info
)
from protein_holography_pytorch.preprocessing_faster.preprocessors.preprocessor_pdbs import PDBPreprocessor
from protein_holography_pytorch.utils.log_config import format
# from protein_holography_pytorch.utils.posterity import get_metadata,record_metadata

from typing import *

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format=format)


def get_structural_info(pdb_file: Union[str, List[str]],
                        padded_length: int=200000,
                        parser: str = 'pyrosetta',
                        relax: bool = False,
                        relax_bb: bool = False):

    """ Get structural info from a single pdb file """

    assert parser in {'pyrosetta', 'biopython'}, f'Parser cannot be {parser}'

    if isinstance(pdb_file, str):
        L = len(pdb_file.split('/')[-1].split('.')[0])
    else:
        L = len(pdb_file[0].split('/')[-1].split('.')[0])
        for i in range(1, len(pdb_file)):
            L = max(L, len(pdb_file[i].split('/')[-1].split('.')[0]))

    dt = np.dtype([
        ('pdb',f'S{L}',()),
        ('atom_names', 'S4', (padded_length)),
        ('elements', 'S1', (padded_length)),
        ('res_ids', f'S{L}', (padded_length, 6)),
        ('coords', 'f4', (padded_length, 3)),
        ('SASAs', 'f4', (padded_length)),
        ('charges', 'f4', (padded_length)),
    ])

    if isinstance(pdb_file, str):
        pdb_file = [pdb_file]
    
    np_protein = np.zeros(shape=(len(pdb_file),), dtype=dt) 

    n = 0
    for i, pdb_file in enumerate(pdb_file):

        si = get_padded_structural_info(pdb_file, padded_length=padded_length, parser=parser, relax=relax, relax_bb=relax_bb)
        if si[0] is None:
            continue

        np_protein[n] = (*si,)
        n += 1

    np_protein.resize((n,))

    return np_protein

def get_structural_info_from_pyrosetta_pose(pose,
                                            padded_length: int=200000,
                                            relax: bool = False,
                                            relax_bb: bool = False):

    si = get_padded_structural_info(pose, padded_length=padded_length, parser='pyrosetta', relax=relax, relax_bb=relax_bb)
    if si[0] is None:
        print('Error processing pose.')
        return None

    dt = np.dtype([
        ('pdb',f'S{len(pose.pdb_info().name())}',()),
        ('atom_names', 'S4', (padded_length)),
        ('elements', 'S1', (padded_length)),
        ('res_ids', f'S{len(pose.pdb_info().name())}', (padded_length, 6)),
        ('coords', 'f4', (padded_length, 3)),
        ('SASAs', 'f4', (padded_length)),
        ('charges', 'f4', (padded_length)),
    ])

    np_protein = np.zeros(shape=(1,), dtype=dt) 
    np_protein[0] = (*si,)

    return np_protein


def get_padded_structural_info(
    pdb_file_or_pose: str,
    padded_length: int=200000,
    parser: str = 'pyrosetta',
    relax: bool = False,
    relax_bb: bool = False) -> Tuple[
    bytes,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:

    try:
        if parser == 'biopython':
            if relax or relax_bb:
                raise Warning("Relaxation not implemented for Biopython parser")
            raise NotImplementedError("Use of Biopython parser not implemented yet")
        elif parser == 'pyrosetta':
            pdb, ragged_structural_info = get_structural_info_from_protein__pyrosetta(pdb_file_or_pose, relax=relax, relax_bb=relax_bb)

        mat_structural_info = pad_structural_info(
            ragged_structural_info,padded_length=padded_length
        )
    except Exception as e:
        logger.error(f"Failed to process {pdb_file_or_pose}")
        logger.error(e)
        return (None,)

    return (pdb, *mat_structural_info)


def get_structural_info_from_dataset(
    pdb_list_file: str,
    pdb_dir: str,
    max_atoms: int,
    relax: bool,
    relax_bb: bool,
    hdf5_out: str,
    output_dataset_name: str,
    parallelism: int,
    compression=LZ4(),
    logging_level=logging.INFO
):
    """
    Parallel processing of pdbs into structural info
    
    Parameters
    ---------
    pdb_list : str
        path to csv file containing list of pdbs, under the column name 'pdb'
    pdb_dir : str
        Path where the pdb files are stored
    max_atoms : int
        Max number of atoms in a protein for padding purposes
    hdf5_out : str
        Path to hdf5 file to write
    parlellism : int
        Number of workers to use
    """

    logger.setLevel(logging_level)

    # metadata = get_metadata()
    
    with open(pdb_list_file, 'r') as f:
        pdb_list = [pdb.strip() for pdb in f.readlines()]

    pdb_list_from_dir = []
    for file in os.listdir(pdb_dir):
        if file.endswith(".pdb"):
            pdb = file.strip('.pdb')
            pdb_list_from_dir.append(pdb)
    
    # filter out pdbs that are not in the directory
    pdb_list = list(set(pdb_list) & set(pdb_list_from_dir))
    
    ds = PDBPreprocessor(pdb_list, pdb_dir)
    bad_neighborhoods = []
    L = np.max([ds.pdb_name_length, 5])
    logger.info(f"Maximum pdb name L = {L}")
    
    dt = np.dtype([
        ('pdb', f'S{L}',()),
        ('atom_names', 'S4', (max_atoms)),
        ('elements', 'S1', (max_atoms)),
        ('res_ids', f'S{L}', (max_atoms,6)),
        ('coords', 'f4', (max_atoms,3)),
        ('SASAs', 'f4', (max_atoms)),
        ('charges', 'f4', (max_atoms)),
    ])
    
    with h5py.File(hdf5_out,'w') as f:
        f.create_dataset(output_dataset_name,
                         shape=(ds.size,),
                         dtype=dt,
                         compression=compression
                         )
        # record_metadata(metadata, f[output_dataset_name])

    with Bar('Processing', max = ds.count(), suffix='%(percent).1f%%') as bar:
        with h5py.File(hdf5_out,'r+') as f:
            n = 0
            for i,structural_info in enumerate(ds.execute(
                    get_padded_structural_info,
                    limit = None,
                    params = {'padded_length': max_atoms, 'relax': relax, 'relax_bb': relax_bb},
                    parallelism = parallelism)):
                if structural_info[0] is None:
                    bar.next()
                    continue
                try:
                    (pdb,atom_names,elements,
                    res_ids,coords,sasas,charges) = (*structural_info,)
                    f[output_dataset_name][n] = (
                        pdb, atom_names, elements,
                        res_ids, coords, sasas, charges
                    )
                    logger.info(f"Wrote to hdf5 for pdb = {pdb}")
                except Exception as e:
                    print(e, file=sys.stderr)
                    bar.next()
                    continue

                n+=1
                bar.next()
            
            print(f"----------> n = {n}")
            f[output_dataset_name].resize((n,)) # resize to account for errors

def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--hdf5_out', type=str,
        help='Output hdf5 filename, where structural info will be stored.',
        required=True
    )
    parser.add_argument(
        '--output_dataset_name', type=str,
        help='Name of the dataset within output_hdf5 where the structural information will be saved. We recommend keeping this set to simply "data".',
        default='data'
    )
    parser.add_argument(
        '--pdb_list_file', type=str,
        help='Path to file containing list of PDB files of interest, one per row.',
        required=True
    )
    parser.add_argument(
        '--pdb_dir', type=str,
        help='directory of pbb files',
        required=True
    )    
    parser.add_argument(
        '--parallelism', type=int,
        help='output file name',
        default=4
    )
    parser.add_argument(
        '--max_atoms', type=int,
        help='max number of atoms per protein for padding purposes',
        default=200000
    )
    parser.add_argument(
        '--relax', action='store_true',
        help='relax protein before processing',
        default=False
    )
    parser.add_argument(
        '--relax_bb', action='store_true',
        help='whether to relax the backbone atoms as well; slower processing but potentially more accurate/meaningful',
        default=False
    )
    parser.add_argument(
        '--logging', type=str,
        help='logging level',
        default="INFO"
    )
    
    args = parser.parse_args()

    # os.environ['NUMEXPR_MAX_THREADS'] = '4' #str(args.parallelism)

    get_structural_info_from_dataset(
        args.pdb_list_file,
        args.pdb_dir,
        args.max_atoms,
        args.relax,
        args.relax_bb,
        args.hdf5_out,
        args.output_dataset_name,
        args.parallelism,
        logging_level=eval(f'logging.{args.logging}')
    )

if __name__ == "__main__":
    start_time=time()
    main()
    print(f"Total time = {time() - start_time:.2f} seconds")
