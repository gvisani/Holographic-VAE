"""Module for extracting structural info from pyrosetta pose"""

import os, sys

from functools import partial
from pathlib import Path
from typing import List,Tuple

import h5py
import numpy as np

## LOCAL ABSOLUTE PYROSETTA PATH
sys.path.append('/gscratch/spe/gvisan01/PyRosetta4.Release.python39.linux.release-335')

import pyrosetta
from pyrosetta.toolbox.extract_coords_pose import pose_coords_as_rows
from pyrosetta.rosetta.core.pose import Pose
from pyrosetta.rosetta.core.id import (
    AtomID,AtomID_Map_double_t,AtomID_Map_bool_t)
from pyrosetta.rosetta.core.scoring import calc_per_atom_sasa
from pyrosetta.rosetta.core.scoring.hbonds import HBondSet
from pyrosetta.rosetta.protocols.moves import DsspMover
from pyrosetta.rosetta.utility import vector1_double

from typing import *


def get_structural_info(pdb_filepath: Union[str, List[str]],
                        max_atoms: int = 200000
                        ) -> np.array:
    '''
    Get structural info from either a single pdb file, or a list of pdbs, using pyrosetta.
    '''

    # init_flags = '-ignore_unrecognized_res 1 -include_current -ex1 -ex2 -ignore_waters 1'
    init_flags = '-ignore_unrecognized_res 1 -include_current -ex1 -ex2 -mute all -include_sugars -ignore_zero_occupancy false -obey_ENDMDL 1'
    pyrosetta.init(init_flags, silent=True)

    dt = np.dtype([
            ('pdb','S50',()),
            ('atom_names', 'S4', (max_atoms)),
            ('elements', 'S1', (max_atoms)),
            ('res_ids', 'S50', (max_atoms, 6)),
            ('coords', 'f4', (max_atoms, 3)),
            ('SASAs', 'f4', (max_atoms)),
            # ('RSAs', 'f4', (max_atoms)),
            ('charges', 'f4', (max_atoms)),
        ])

    if isinstance(pdb_filepath, str):
        pdb_filepath = [pdb_filepath]
    
    np_protein = np.zeros(shape=(len(pdb_filepath)), dtype=dt) 

    for i, pdb_file in enumerate(pdb_filepath):
        pose = pyrosetta.pose_from_pdb(pdb_file)

        si = get_padded_structural_info(pose, padded_length=max_atoms)
        np_protein[i] = (*si,)

    return np_protein


################################################################################


def get_padded_structural_info(
    pose: Pose, padded_length: int=200000) -> Tuple[
    bytes,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """
    Extract structural info used for holographic projection from PyRosetta pose.
    
    Parameters
    ----------
    pose : pyrosetta.rosetta.core.pose.Pose
        Pose created by PyRosetta from pdb file
        
    Returns
    -------
    tuple of (bytes, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, 
              np.ndarray)
        The entries in the tuple are
            bytes encoding the pdb name string
            bytes array encoding the atom names of shape [max_atoms]
            bytes array encoding the elements of shape [max_atoms]
            bytes array encoding the residue ids of shape [max_atoms,6]
            float array of shape [max_atoms,3] representing the 3D Cartesian 
              coordinates of each atom
            float array of shape [max_atoms] storing the SASA of each atom
            float array of shape [max_atoms] storing the partial charge of each atom
    """

    pdb,ragged_structural_info = get_structural_info_from_protein(pose)
    mat_structural_info = pad_structural_info(
        ragged_structural_info,padded_length=padded_length
    )

    return (pdb,*mat_structural_info)

def get_pose_residue_number(
    pose: Pose, 
    chain: str, 
    resnum: int, 
    icode: str=' '
) -> int:
    """Translate pdb residue id to pyrosetta index"""
    return pose.pdb_info().pdb2pose(chain, resnum, icode)

def get_pdb_residue_info(
    pose : Pose, 
    resnum : int
) -> Tuple[str, int, str]:
    """Translate pyrosetta index to pdb residue id"""
    pi = pose.pdb_info()
    return (pi.chain(resnum), pi.number(resnum), pi.icode(resnum))

def calculate_sasa(
    pose : Pose,
    probe_radius : float=1.4
) -> AtomID_Map_double_t:
    """Calculate SASA for a pose"""
    # pyrosetta structures for returning of sasa information
    all_atoms = AtomID_Map_bool_t()
    atom_sasa = AtomID_Map_double_t()
    rsd_sasa = vector1_double()
    
    # use pyrosetta to calculate SASA per atom
    calc_per_atom_sasa(
        pose,
        atom_sasa,
        rsd_sasa,
        probe_radius
    )
    
    return atom_sasa

def get_hb_counts(
    hbond_set: HBondSet,
    i: int
):
    """
    Classifies a pose's h-bonds by main- and side-chain linkages
    
    Parameters
    ----------
    hbond_set : 
        The h-bond object from pyrosetta
    i : int
       
    Returns
    -------
    np.ndarray
        Float array of shape [8] where each entry is the number of
        h-bonds where the central residue and the partner are categorized 
        according to the donor/accceptor role and the backbone (bb) vs. 
        side-chain (sc) location of the bond. Specifically, the array is 
        organized as
            central|partner
            ---------------
            acc-bb  don-bb
            don-bb  acc-bb
            acc-bb  don-sc
            don-bb  acc-sc
            acc-sc  don-bb
            don-sc  acc-bb
            acc-sc  don-sc
            don-sc  acc-sc
    """
    counts = np.zeros(8,dtype=int)
    for hb in hbond_set.residue_hbonds(i):
        ctrl_don = hb.don_res() == i
        if ctrl_don:
            ctrl_side = not hb.don_hatm_is_backbone()
            nb_side = not hb.acc_atm_is_backbone()
        else:
            ctrl_side = not hb.acc_atm_is_backbone()
            nb_side = not hb.don_hatm_is_backbone()
        counts[4*ctrl_side + 2*nb_side + 1*ctrl_don] += 1
    return counts

def get_structural_info_from_protein(pose : Pose) -> Tuple[
    str,
    Tuple[
        np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray
    ]
]:
    """
    Extract structural information from pyrosetta pose
    
    Parameters
    ----------
    pose : pyrosetta.rosetta.core.pose.Pose
        The pose created for the protein of interest
      
    Returns
    -------
    nested tuple of (bytes, (np.ndarray, np.ndarray, np.ndarray, np.ndarray,
      np.ndarray,np.ndarray)
        This nested tuple contains the pdb name followed by arrays containing
        the atom names, elements, residue ids, coordinates, SASAs, and charges 
        for each atom in the protein.
    """

    # from .biophysics import MAXASA_TABLE

    # lists for each type of information to obtain
    atom_names = []
    elements = []
    sasas = []
    # rsas = []
    coords = []
    charges = []
    res_ids = []
    
    k = 0
    
    # extract secondary structure for use in res ids
    DSSP = DsspMover()
    DSSP.apply(pose)
      
    # extract physico-chemical information
    atom_sasa = calculate_sasa(pose)
    coords_rows = pose_coords_as_rows(pose)
    
    pi = pose.pdb_info()
    pdb = Path(pi.name()).stem.encode()
    
    # get structural info from each residue in the protein
    for i in range(1,pose.size()+1):
        
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
            # rsa = sasa / MAXASA_TABLE['theoretical'][aa]
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
            ], dtype='S6')
            
            atom_names.append(atom_name)
            elements.append(element)
            res_ids.append(res_id)
            coords.append(curr_coords)
            sasas.append(sasa)
            # rsas.append(rsa)
            charges.append(charge)
            
            k += 1
            
    atom_names = np.array(atom_names,dtype='|S4')
    elements = np.array(elements,dtype='S1')
    sasas = np.array(sasas)
    # rsas = np.array(rsas)
    coords = np.array(coords)
    charges = np.array(charges)
    res_ids = np.array(res_ids)
    
    return pdb,(atom_names,elements,res_ids,coords,sasas,charges) #,sasas,rsas,charges)

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

    # check that the padding is large enough to accomodate the data
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
    