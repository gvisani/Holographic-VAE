"""Module for extracting structural info from pyrosetta pose"""

import os, sys

from functools import partial
from pathlib import Path
from typing import List,Tuple

import h5py
import numpy as np

from typing import *


def get_structural_info(pdb_filepath: Union[str, List[str]],
                        max_atoms: int = 200000,
                        parser: str = 'biopython'
                        ) -> np.array:
    '''
    Get structural info from either a single pdb file, or a list of pdbs, using pyrosetta.
    '''

    assert parser in {'pyrosetta', 'biopython'}

    dt = np.dtype([
            ('pdb','S50',()),
            ('atom_names', 'S4', (max_atoms)),
            ('elements', 'S1', (max_atoms)),
            ('res_ids', 'S50', (max_atoms, 6)),
            ('coords', 'f4', (max_atoms, 3)),
            # ('SASAs', 'f4', (max_atoms)),
            # ('RSAs', 'f4', (max_atoms)),
            # ('charges', 'f4', (max_atoms)),
        ])

    if isinstance(pdb_filepath, str):
        pdb_filepath = [pdb_filepath]
    
    np_protein = np.zeros(shape=(len(pdb_filepath)), dtype=dt) 

    for i, pdb_file in enumerate(pdb_filepath):

        si = get_padded_structural_info(pdb_file, padded_length=max_atoms, parser=parser)
        np_protein[i] = (*si,)

    return np_protein


################################################################################


def get_padded_structural_info(
    pdb_file: str, padded_length: int=200000, parser: str = 'biopython') -> Tuple[
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

    if parser == 'biopython':
        pdb, ragged_structural_info = get_structural_info_from_protein__biopython(pdb_file)
    elif parser == 'pyrosetta':
        pdb, ragged_structural_info = get_structural_info_from_protein__pyrosetta(pdb_file)

    mat_structural_info = pad_structural_info(
        ragged_structural_info,padded_length=padded_length
    )

    return (pdb, *mat_structural_info)


def get_structural_info_from_protein__pyrosetta(pdb_file : str) -> Tuple[
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

    import pyrosetta
    from pyrosetta.toolbox.extract_coords_pose import pose_coords_as_rows
    from pyrosetta.rosetta.core.id import AtomID
    from pyrosetta.rosetta.protocols.moves import DsspMover

    from .pyrosetta_utils import calculate_sasa

    # init_flags = '-ignore_unrecognized_res 1 -include_current -ex1 -ex2 -ignore_waters 1'
    init_flags = '-ignore_unrecognized_res 1 -include_current -ex1 -ex2 -mute all -include_sugars -ignore_zero_occupancy false -obey_ENDMDL 1'

    pyrosetta.init(init_flags, silent=True)

    pose = pyrosetta.pose_from_pdb(pdb_file)

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
    
    return pdb,(atom_names,elements,res_ids,coords) #,sasas,charges)


def get_structural_info_from_protein__biopython(
    pdb_file : str,
    remove_nonwater_hetero: bool = False,
    remove_waters: bool = True,
    ):
    
    '''
    atom full id:
        - (PDB, model_num, chain, (hetero_flag, resnum, insertion_code), (atom_name, disorder_altloc))
    
    By default, biopyton selects only atoms with the highest occupancy, thus behaving like pyrosetta does with the flag "-ignore_zero_occupancy false"
    '''

    from Bio.PDB import PDBParser
    parser = PDBParser()

    structure = parser.get_structure(pdb_file[:-4], pdb_file)

    
    aa_to_one_letter = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
                        'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
                        'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
                        'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER':'S',
                        'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}

    # assume only one model is present in the structure
    models = list(structure.get_models())
    assert len(models) == 1

    # assume the pdb name was provided as id to create the structure
    pdb = structure.get_id()

    # lists for each type of information to obtain
    atom_names = []
    elements = []
    coords = []
    res_ids = []
    
    k = 0
    
    def pad_for_consistency(string):
        return (' ' + string).ljust(4, ' ')
    
    # get structural info from each residue in the protein
    for atom in structure.get_atoms():

        atom_full_id = atom.get_full_id()
        
        if remove_waters and atom_full_id[3][0] == 'W':
            continue
        
        if remove_nonwater_hetero and atom_full_id[3][0] not in {' ' 'W'}:
            continue

        chain = atom_full_id[2]
        resnum = atom_full_id[3][1]
        icode = atom_full_id[3][2]
        atom_name = pad_for_consistency(atom.get_name())
        element = atom.element
        coord = atom.get_coord()

        aa = atom.get_parent().resname
        if aa in aa_to_one_letter:
            aa = aa_to_one_letter[aa]

        res_id = np.array([aa,pdb,chain,resnum,icode,'null'],dtype='S5') # adding 'null' in place of secondary structure for compatibility
        
        atom_names.append(atom_name)
        elements.append(element)
        res_ids.append(res_id)
        coords.append(coord)
        
        k += 1
            
    atom_names = np.array(atom_names,dtype='|S4')
    elements = np.array(elements,dtype='S1')
    coords = np.array(coords)
    res_ids = np.array(res_ids)
    
    return pdb,(atom_names,elements,res_ids,coords)

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
    