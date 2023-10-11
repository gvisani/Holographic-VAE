"""Module for extracting structural info from pyrosetta pose"""

from functools import partial
import logging
from pathlib import Path
import sys
from typing import List,Tuple

import h5py
import numpy as np

from typing import *


## un-comment the following three lines for faster, bulk processing with pyrosetta, and comment out the ones in the get_structural_info_from_protein__pyrosetta() function below
import pyrosetta
init_flags = '-ignore_unrecognized_res 1 -include_current -ex1 -ex2 -mute all -include_sugars -ignore_zero_occupancy false -obey_ENDMDL 1'
pyrosetta.init(init_flags, silent=True)


def get_structural_info_from_protein__pyrosetta(pdb_file: Union[str, pyrosetta.rosetta.core.pose.Pose],
                                                relax: bool = False,
                                                relax_bb: bool = False) -> Tuple[
    str,
    Tuple[
        np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray
    ]
]:
    """
    Extract structural information from pyrosetta pose
    
    Parameters
    ----------
    pdb_file : str or pyrosetta.rosetta.core.pose.Pose
        The pdb_file or the pyrosetta pose created for the protein of interest
      
    Returns
    -------
    nested tuple of (bytes, (np.ndarray, np.ndarray, np.ndarray, np.ndarray,
      np.ndarray,np.ndarray)
        This nested tuple contains the pdb name followed by arrays containing
        the atom names, elements, residue ids, coordinates, SASAs, and charges 
        for each atom in the protein.
    """

    ## comment out these three lines for faster, bulk processing with pyrosetta, and uncomment the lines at the top of the script
    # import pyrosetta
    # init_flags = '-ignore_unrecognized_res 1 -include_current -ex1 -ex2 -mute all -include_sugars -ignore_zero_occupancy false -obey_ENDMDL 1'
    # pyrosetta.init(init_flags, silent=True)

    from pyrosetta.toolbox.extract_coords_pose import pose_coords_as_rows
    from pyrosetta.rosetta.core.id import AtomID
    from pyrosetta.rosetta.protocols.moves import DsspMover

    from protein_holography_pytorch.preprocessing_faster.utils.pyrosetta import calculate_sasa

    from protein_holography_pytorch.utils.pyrosetta_utils import fastrelax_full_pose

    if isinstance(pdb_file, str):
        logging.debug(f"pdb name in protein routine {pdb_file.split('/')[-1].strip('.pdb')} - start")
        pose = pyrosetta.pose_from_pdb(pdb_file)
    else:
        pose = pdb_file

    # lists for each type of information to obtain
    atom_names = []
    elements = []
    sasas = []
    coords = []
    charges = []
    res_ids = []
    
    k = 0
    
    # extract secondary structure for use in res ids
    DSSP = DsspMover()
    DSSP.apply(pose)

    # note that we're relaxing after computing sec-struct, just so that sec-struct computation is not messed up whenever we relax the backbone
    if relax:
        logging.debug(f'relaxing structure, relax_bb = {relax_bb}')
        fastrelax_full_pose(pose, pyrosetta.create_score_function('ref2015_cart.wts'), relax_backbone=relax_bb)
      
    # extract physico-chemical information
    atom_sasa = calculate_sasa(pose)
    coords_rows = pose_coords_as_rows(pose)
    
    pi = pose.pdb_info()
    pdb = pi.name().split('/')[-1].strip('.pdb')
    L = len(pdb)

    logging.debug(f"pdb name in protein routine {pdb} - successfully loaded pdb into pyrosetta")

    # get structural info from each residue in the protein
    for i in range(1, pose.size()+1):
        
        # these data will form the residue id
        aa = pose.sequence()[i-1]
        chain = pi.chain(i)
        resnum = str(pi.number(i)).encode()
        icode = pi.icode(i).encode()
        ss = pose.secstruct(i)
        
        ## optional info to include in residue ids if analysis merits it
        ## - hbond info
        ## - chi1 angle
        #hbond_set = pose.get_hbonds()
        #chi1 = b''
        #print(aa)
        #if aa not in ['G','A','Z']:
        #    try:
        #        chi1 = str(pose.chi(1,i)).encode()
        #    except:
        #        print(pdb,aa,chain,resnum)
        #        #print(chi1)
        
        for j in range(1,len(pose.residue(i).atoms())+1):

            atom_name = pose.residue_type(i).atom_name(j)
            idx = pose.residue(i).atom_index(atom_name)
            atom_id = (AtomID(idx,i))
            element = pose.residue_type(i).element(j).name
            sasa = atom_sasa.get(atom_id)
            curr_coords = coords_rows[k]
            charge = pose.residue_type(i).atom_charge(j)
            #hb_counts = get_hb_counts(hbond_set,i)
            
            res_id = np.array([
                aa,
                pdb,
                chain,
                resnum,
                icode,
                ss,
                #*hb_counts,
                #chi1
            ], dtype=f'S{L}')
            
            atom_names.append(atom_name)
            elements.append(element)
            res_ids.append(res_id)
            coords.append(curr_coords)
            sasas.append(sasa)
            charges.append(charge)
            
            k += 1
            
    atom_names = np.array(atom_names,dtype='|S4')
    elements = np.array(elements,dtype='S1')
    sasas = np.array(sasas)
    coords = np.array(coords)
    charges = np.array(charges)
    res_ids = np.array(res_ids)
    
    return pdb,(atom_names,elements,res_ids,coords,sasas,charges)

# given a matrix, pad it with empty array
def pad(
    arr: np.ndarray,
    padded_length: int=100
) -> np.ndarray:
    """
    Pad an array long axis 0
    
    Parameters
    ----------
    arr : np.ndarray
    padded_length : int

    Returns
    -------
    np.ndarray
    """
    # get dtype of input array
    dt = arr.dtype

    # shape of sub arrays and first dimension (to be padded)
    shape = arr.shape[1:]
    orig_length = arr.shape[0]

    # check that the padding is large enough to accomdate the data
    if padded_length < orig_length:
        print('Error: Padded length of {}'.format(padded_length),
              'is smaller than original length of array {}'.format(orig_length))

    # create padded array
    padded_shape = (padded_length,*shape)
    mat_arr = np.zeros(padded_shape, dtype=dt)

    # add data to padded array
    mat_arr[:orig_length] = np.array(arr)
    
    return mat_arr

def pad_structural_info(
    ragged_structure: Tuple[np.ndarray, ...],
    padded_length: int=100
) -> List[np.ndarray]:
    """Pad structural into arrays"""
    pad_custom = partial(pad,padded_length=padded_length)
    mat_structure = list(map(pad_custom,ragged_structure))

    return mat_structure
